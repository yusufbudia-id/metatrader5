import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
import sqlite3
from datetime import datetime, timedelta, timezone 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore') 

# ==========================================
# 1. UNIVERSES & PORTFOLIO CONFIGURATION
# ==========================================
UNIVERSE_CONFIG = {
    "FX": {
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        "max_bars": 20, "sl_mult": 1.5, "tp_mult": 2.25, "depth": 4, "max_spread": 25
    },
    "METAL": {
        "symbols": ["XAUUSD", "GOLD"],
        "max_bars": 12, "sl_mult": 2.0, "tp_mult": 4.0, "depth": 5, "max_spread": 400
    },
    "CRYPTO": {
        "symbols": ["BTCUSD"],
        "max_bars": 8, "sl_mult": 3.0, "tp_mult": 6.0, "depth": 6, "max_spread": 5000
    }
}

MAGIC_NUMBER = 999
BASE_RISK_PERCENT = 0.01
MAX_USD_EXPOSURE = 2 
MAX_PORTFOLIO_RISK = 0.03 
CORRELATION_THRESHOLD = 0.75 
SLIPPAGE_POINTS = 300 
MODEL_MAX_AGE = 18 * 3600  
PSI_DRIFT_TRIGGER = 0.12   
MIN_TRADES_FOR_RETRAIN = 25 
DECISION_LOG_FILE = "quant_v17_paradigm_log.csv"

if not mt5.initialize(): quit()

def resolve_symbol(sym_name):
    for s in mt5.symbols_get():
        if s.name.upper().startswith(sym_name.upper()): return s.name
    return sym_name

models_vault = {}
consecutive_losses = 0
last_trade_time = {}
disabled_symbols = {} 
global_corr_matrix = pd.DataFrame() 
last_corr_update = datetime.min 
spread_history = {} 
exit_cooldown = {} 
trade_context = {} 
model_trade_count = {} 

# ==========================================
# 2. TELEMETRY, HEALTH, AND DATABASES
# ==========================================
db_conn = sqlite3.connect("quant_production_log.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TEXT, symbol TEXT, action TEXT, raw_pred INTEGER, 
        conf REAL, threshold REAL, vol_ratio REAL, spread REAL, 
        status TEXT, ticket INTEGER, expected_edge REAL, realized_pnl REAL
    )
''')
db_conn.commit()

def log_decision(symbol, action, raw_pred, conf, threshold, vol_ratio, spread, status, ticket=None, expected_edge=0.0, realized_pnl=0.0):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    db_cursor.execute('''
        INSERT INTO trade_decisions (time, symbol, action, raw_pred, conf, threshold, vol_ratio, spread, status, ticket, expected_edge, realized_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, symbol, action, raw_pred, conf, threshold, vol_ratio, spread, status, ticket, expected_edge, realized_pnl))
    db_conn.commit()

def sync_db_pnl():
    rows = db_cursor.execute("SELECT ticket FROM trade_decisions WHERE status='EXECUTED' AND realized_pnl=0.0").fetchall()
    if not rows: return
    hist = mt5.history_deals_get(datetime.now() - timedelta(days=7), datetime.now())
    if not hist: return
    hist_dict = {d.position_id: d for d in hist if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)}
    for row in rows:
        ticket = row[0]
        if ticket in hist_dict:
            db_cursor.execute("UPDATE trade_decisions SET realized_pnl=? WHERE ticket=?", (hist_dict[ticket].profit, ticket))
    db_conn.commit()

def check_calibration_drift(sym):
    sync_db_pnl() 
    rows = db_cursor.execute("""
        SELECT expected_edge, realized_pnl
        FROM trade_decisions
        WHERE symbol=? AND status='EXECUTED' AND realized_pnl != 0.0
        ORDER BY id DESC LIMIT 40
    """, (sym,)).fetchall()

    if len(rows) < 20: return True

    df = pd.DataFrame(rows, columns=['edge', 'pnl'])
    if df['edge'].std() == 0 or df['pnl'].std() == 0: return True
    corr = df['edge'].corr(df['pnl'])
    
    if pd.isna(corr) or corr < 0.05:
        print(f"[{sym}] Calibration breakdown detected (Corr: {corr:.2f}). Suspending model.")
        disabled_symbols[sym] = datetime.now() + timedelta(hours=6)
        return False
    return True

def heartbeat():
    with open("heartbeat.txt", "w") as f: f.write(str(datetime.now()))

def account_kill_switch():
    acc = mt5.account_info()
    if not acc: return True 
    hist = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
    if hist:
        pnl = sum(d.profit for d in hist if d.magic == MAGIC_NUMBER)
        if pnl < -0.05 * acc.equity:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] !!! GLOBAL KILL SWITCH !!! Drawdown > 5%")
            return True
    return False

def get_equity_regime():
    acc = mt5.account_info()
    if not acc: return 1.0
    hist = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
    if not hist: return 1.0
    deals = sorted([d for d in hist if d.magic == MAGIC_NUMBER and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)], key=lambda x: x.time, reverse=True)[:30]
    if len(deals) < 10: return 1.0
    pnl = sum(d.profit for d in deals)
    return np.clip(1 + (pnl / acc.equity), 0.7, 1.2)

def update_correlation_matrix():
    global global_corr_matrix
    returns = {}
    for sym in models_vault.keys():
        df = get_safe_rates(sym, mt5.TIMEFRAME_M15, 1500) 
        if df is not None:
            ret = df['close'].pct_change()
            vol = ret.rolling(48).std()
            returns[sym] = (ret / vol).ewm(span=48).mean().dropna()
    if returns:
        global_corr_matrix = pd.DataFrame(returns).corr().fillna(0.0)

def get_position_risk_money(p):
    info = mt5.symbol_info(p.symbol)
    if not info or info.point == 0: return 0.0
    if p.sl == 0.0: 
        acc = mt5.account_info()
        return acc.equity * BASE_RISK_PERCENT if acc else 0.0
    return (abs(p.price_open - p.sl) / info.point) * info.trade_tick_value * p.volume

def get_current_portfolio_risk():
    positions = mt5.positions_get()
    return sum(get_position_risk_money(p) for p in positions if p.magic == MAGIC_NUMBER) if positions else 0.0

def get_usd_impact(symbol, action):
    base, quote = symbol[:3], symbol[3:6]
    side = 1 if action == "BUY" else -1
    return side if base == "USD" else (-side if quote == "USD" else 0)

def get_usd_net_exposure():
    positions = mt5.positions_get()
    return sum(get_usd_impact(p.symbol, "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL") for p in positions) if positions else 0

def get_live_outcome_feature(sym, current_time):
    history = mt5.history_deals_get(current_time - timedelta(days=5), current_time)
    if not history: return 0
    valid = [d for d in history if d.symbol == sym and d.magic == MAGIC_NUMBER and d.time <= current_time.timestamp() and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]
    if not valid: return 0
    last = max(valid, key=lambda x: x.time)
    return 1 if last.profit > 0 else -1

def get_live_edge(sym, current_time):
    history = mt5.history_deals_get(current_time - timedelta(days=7), current_time)
    if not history: return 0
    valid = [d for d in history if d.symbol == sym and d.magic == MAGIC_NUMBER and d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT)]
    wins = sum(1 for d in valid if d.profit > 0)
    losses = len(valid) - wins
    return (wins - losses) / len(valid) if valid else 0

def calculate_psi(expected, actual, bins=10):
    min_val, max_val = min(np.min(expected), np.min(actual)), max(np.max(expected), np.max(actual))
    if min_val == max_val: max_val += 1e-6
    breakpoints = np.linspace(min_val, max_val, bins + 1)
    expected_perc = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)
    return np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

def get_safe_rates(sym, timeframe, count):
    rates = mt5.copy_rates_from_pos(sym, timeframe, 0, count)
    if rates is None or len(rates) < count: return None
    df = pd.DataFrame(rates)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_14'] = df['ATR_14'].bfill().ffill()
    return df

def calc_entropy(x):
    hist = np.histogram(x, bins=5, density=True)[0]
    return -np.sum(hist * np.log(hist + 1e-6))

def update_loss_streak():
    global consecutive_losses
    history = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
    if not history: return
    sorted_deals = sorted(history, key=lambda x: x.time, reverse=True)
    temp_losses = 0
    for deal in sorted_deals:
        if deal.magic == MAGIC_NUMBER and deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
            if deal.profit < 0: temp_losses += 1
            else: break
    consecutive_losses = temp_losses

# ==========================================
# 3. ADVANCED AI ENGINE 
# ==========================================
def validate_model(new_model, old_model):
    return new_model["threshold"] >= old_model["threshold"] * 0.95

def create_dynamic_cost_labels(df, sl_mult, tp_mult, max_bars):
    labels = np.zeros(len(df))
    close, high, low, atr = df['close'].values, df['high'].values, df['low'].values, df['ATR_14'].values
    spread_proxy, vol_accel = df['spread_proxy'].values, df['vol_accel'].values

    for i in range(len(df) - max_bars):
        if np.isnan(atr[i]) or np.isnan(spread_proxy[i]) or np.isnan(vol_accel[i]): continue
        cost = (spread_proxy[i] / (atr[i] + 1e-6)) * atr[i] * (1 + 0.5 * max(0, vol_accel[i]))
        
        long_entry, short_entry = close[i] + cost, close[i] - cost
        long_tp, long_sl = long_entry + (tp_mult * atr[i]), long_entry - (sl_mult * atr[i])
        short_tp, short_sl = short_entry - (tp_mult * atr[i]), short_entry + (sl_mult * atr[i])

        hit_long_tp, hit_long_sl, hit_short_tp, hit_short_sl = False, False, False, False

        for j in range(1, max_bars + 1):
            if not hit_long_tp and not hit_long_sl:
                if high[i+j] >= long_tp: hit_long_tp = True
                elif low[i+j] <= long_sl: hit_long_sl = True
            if not hit_short_tp and not hit_short_sl:
                if low[i+j] <= short_tp: hit_short_tp = True
                elif high[i+j] >= short_sl: hit_short_sl = True

        if hit_long_tp and not hit_long_sl: labels[i] = 1
        elif hit_short_tp and not hit_short_sl: labels[i] = -1

    return labels

def train_model(sym, config):
    df = get_safe_rates(sym, mt5.TIMEFRAME_M15, 6500)
    if df is None: return None
    
    df.ta.ema(length=50, append=True)
    df['hl_range'] = df['high'] - df['low']
    df['spread_proxy'] = df['hl_range'].rolling(5).median()
    df['spread_norm'] = df['spread_proxy'] / df['ATR_14']
    df['vol_ratio'] = df['ATR_14'] / df['ATR_14'].rolling(200).mean()
    df['dist_ema_z'] = (df['close'] - df['EMA_50']) / df['ATR_14']
    df['roc_10'] = df['close'].pct_change(10)
    df['trend_strength'] = abs(df['EMA_50'].pct_change(5)) / df['ATR_14']
    df['range_compress'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['ATR_14']
    df['clv'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    df['vol_accel'] = df['ATR_14'].pct_change(5)
    
    df['kurtosis_50'] = df['close'].pct_change().rolling(50).kurt()
    df['skew_50'] = df['close'].pct_change().rolling(50).skew()
    df['realized_vol'] = df['close'].pct_change().rolling(30).std()
    df['vol_of_vol'] = df['realized_vol'].rolling(30).std()
    df['trend_persistence'] = abs(df['EMA_50'].pct_change(10)) / (df['realized_vol'] + 1e-6)
    
    df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['session'] = np.select([df['hour'] < 7, df['hour'] < 13, df['hour'] < 21], [0, 1, 2], default=3)
    
    df['Target'] = create_dynamic_cost_labels(df, config['sl_mult'], config['tp_mult'], config['max_bars'])
    df['last_setup_outcome'] = df['Target'].shift(config['max_bars']).fillna(0)
    
    raw_live_edge = df['last_setup_outcome'].rolling(100).mean().shift(24).fillna(0)
    df['live_edge'] = np.tanh(raw_live_edge * 1.5)
    
    features = [
        'dist_ema_z', 'vol_ratio', 'roc_10', 'trend_strength', 'spread_norm', 
        'range_compress', 'clv', 'vol_accel', 'session', 'last_setup_outcome',
        'kurtosis_50', 'skew_50', 'realized_vol', 'vol_of_vol', 'trend_persistence', 'live_edge'
    ]
    df[features] = df[features].shift(1) 
    df.dropna(inplace=True)
    if df['Target'].abs().sum() < 200: return None

    train_df = df.iloc[-3500:-500]
    val_df = df.iloc[-500:]
    
    X_train, y_train = train_df[features], train_df['Target']
    X_val = val_df[features]
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    base_model = GradientBoostingClassifier(n_estimators=100, max_depth=config['depth'], random_state=42)
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train_scaled, y_train)
    
    val_probs = calibrated_model.predict_proba(X_val_scaled)
    new_threshold = np.quantile(np.max(val_probs, axis=1), 0.70)
    
    threshold = (0.7 * config["prev_threshold"] + 0.3 * new_threshold) if "prev_threshold" in config else new_threshold
    config["prev_threshold"] = threshold
    
    train_sample = X_train.tail(1000).to_dict(orient='list')
    print(f"[{sym}] V17 Trained | Thresh: {threshold:.4f}")
    
    return {
        "model": calibrated_model, "scaler": scaler, "feature_order": features, 
        "threshold": threshold, "train_sample": train_sample,
        "config": config, "last_train": datetime.now()
    }

# ==========================================
# 4. PORTFOLIO GATE & EXECUTION
# ==========================================
def check_model_health(sym):
    history = mt5.history_deals_get(datetime.now() - timedelta(days=30), datetime.now())
    if not history: return True
    sym_deals = [d for d in history if d.symbol == sym and d.magic == MAGIC_NUMBER and d.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]]
    sym_deals = sorted(sym_deals, key=lambda x: x.time, reverse=True)[:30]
    
    recent5 = sym_deals[:5]
    if len(recent5) == 5 and len(sym_deals) > 5:
        avg_loss = np.mean([abs(d.profit) for d in sym_deals])
        if sum(d.profit for d in recent5) < -1.5 * avg_loss:
            print(f"[{sym}] FAST FAILURE DETECTED. Suspending Model.")
            disabled_symbols[sym] = datetime.now() + timedelta(hours=4) 
            return False

    if len(sym_deals) >= 10:
        winrate = sum(1 for d in sym_deals if d.profit > 0) / len(sym_deals)
        if winrate < 0.40:
            print(f"!!! KILL SWITCH for {sym} !!! Winrate: {winrate:.2f}")
            disabled_symbols[sym] = datetime.now() + timedelta(hours=24) 
            return False
    return True

def manage_early_exits(sym, m_data, X_scaled, adaptive_threshold, atr_now):
    positions = mt5.positions_get(symbol=sym)
    if not positions: return
    
    probs = m_data['model'].predict_proba(X_scaled)[0]
    raw_pred = int(m_data['model'].classes_[np.argmax(probs)])
    conf = np.max(probs)
    now = datetime.now()

    for p in positions:
        if p.magic != MAGIC_NUMBER: continue
        if p.ticket in exit_cooldown and (now - exit_cooldown[p.ticket]).total_seconds() < 60: continue
            
        profit_val = (p.price_current - p.price_open) if p.type == mt5.POSITION_TYPE_BUY else (p.price_open - p.price_current)
        atr_entry = trade_context.get(p.ticket, {}).get("atr_entry", atr_now)
        profit_points_r = profit_val / (atr_entry + 1e-6)
        
        # FIX ISSUE 2: Edge Decay Validation
        entry_edge = trade_context.get(p.ticket, {}).get("expected_edge", 0.01)
        current_edge = conf - adaptive_threshold
        edge_decay = current_edge / (entry_edge + 1e-6)
        
        should_close = False
        if p.type == mt5.POSITION_TYPE_BUY and (raw_pred == -1 or edge_decay < 0.4) and profit_points_r < 0.5:
            should_close = True
        elif p.type == mt5.POSITION_TYPE_SELL and (raw_pred == 1 or edge_decay < 0.4) and profit_points_r < 0.5:
            should_close = True
            
        if should_close:
            res = close_position(p.ticket, sym, p.type, p.volume)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                exit_cooldown[p.ticket] = now
                db_cursor.execute("UPDATE trade_decisions SET realized_pnl=? WHERE ticket=?", (profit_points_r, p.ticket))
                db_conn.commit()
                print(f"[{sym}] EARLY EXIT DONE | Ticket: {p.ticket} | Edge Decay: {edge_decay:.2f} | PnL: {profit_points_r:.2f}R")

def close_position(ticket, sym, action_type, lot):
    tick = mt5.symbol_info_tick(sym)
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "position": ticket, "symbol": sym, "volume": lot,
        "type": mt5.ORDER_TYPE_SELL if action_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if action_type == mt5.POSITION_TYPE_BUY else tick.ask,
        "deviation": SLIPPAGE_POINTS, "magic": MAGIC_NUMBER,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

def execute_trade(sym, action, raw_pred, conf, m_data, adaptive_threshold, vol_ratio, spread_points, atr_now):
    if mt5.positions_get(symbol=sym): return
    if sym in last_trade_time and (datetime.now() - last_trade_time[sym]) < timedelta(minutes=30): return

    open_positions = mt5.positions_get()
    if open_positions and not global_corr_matrix.empty and sym in global_corr_matrix.columns:
        for p in open_positions:
            if p.symbol in global_corr_matrix.index:
                corr = global_corr_matrix.loc[sym, p.symbol]
                if pd.isna(corr): continue 
                if abs(corr) > CORRELATION_THRESHOLD:
                    log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, f"REJECT_CORR_{p.symbol}")
                    return

    acc = mt5.account_info()
    info = mt5.symbol_info(sym)
    sl_dist_points = (m_data['config']['sl_mult'] * atr_now) / info.point
    
    # FIX ISSUE 3: Logistic Risk Scaling (Smooth Saturation)
    edge_factor = (conf - adaptive_threshold) / (1 - adaptive_threshold) if adaptive_threshold < 1 else 0
    risk_boost = 1 / (1 + np.exp(-4 * (edge_factor - 0.3)))
    risk_percent = BASE_RISK_PERCENT * (1 + 1.5 * risk_boost)
    risk_percent = np.clip(risk_percent, BASE_RISK_PERCENT, BASE_RISK_PERCENT * 2.5)
    
    target_vol = 1.0
    vol_adjustment = target_vol / np.clip(vol_ratio, 0.5, 2.0)
    risk_percent *= vol_adjustment
    risk_percent *= get_equity_regime()
    
    new_trade_risk_money = acc.equity * risk_percent
    current_portfolio_risk = get_current_portfolio_risk()
    
    if (current_portfolio_risk + new_trade_risk_money) > (acc.equity * MAX_PORTFOLIO_RISK):
        log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, "REJECT_PORTFOLIO_RISK_CAP")
        return

    pip_value = info.trade_tick_value / info.trade_tick_size 
    raw_lot = new_trade_risk_money / (sl_dist_points * pip_value)
    lot = max(info.volume_min, min(raw_lot, info.volume_max))
    lot = round(lot / info.volume_step) * info.volume_step

    tick = mt5.symbol_info_tick(sym)
    price = tick.ask if action == "BUY" else tick.bid
    sl = price - (sl_dist_points * info.point) if action == "BUY" else price + (sl_dist_points * info.point)
    tp = price + (m_data['config']['tp_mult'] * atr_now) if action == "BUY" else price - (m_data['config']['tp_mult'] * atr_now)

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price, "sl": sl, "tp": tp, "deviation": SLIPPAGE_POINTS,
        "magic": MAGIC_NUMBER, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(request)
    if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
        time.sleep(1)
        res = mt5.order_send(request)
        
    time.sleep(0.5) 
    verify_pos = mt5.positions_get(symbol=sym)
    
    if res and res.retcode == mt5.TRADE_RETCODE_DONE and verify_pos:
        last_trade_time[sym] = datetime.now()
        expected_edge = conf - adaptive_threshold
        trade_context[res.order] = {"atr_entry": atr_now, "expected_edge": expected_edge} 
        model_trade_count[sym] = model_trade_count.get(sym, 0) + 1 
        
        log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, "EXECUTED", res.order, expected_edge, 0.0)
        print(f"[{sym}] {action} VERIFIED | Lot: {lot} | Expected Edge: {expected_edge:.2f}")
    else:
        status = f"DESYNC_FAILED_{res.retcode if res else 'TIMEOUT'}"
        log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, status)

# ==========================================
# 5. MAIN EVENT DAEMON
# ==========================================
print("Initializing Apex Quant Models...")
for cat, cfg in UNIVERSE_CONFIG.items():
    for s_name in cfg['symbols']:
        real_sym = resolve_symbol(s_name)
        model_trade_count[real_sym] = 0
        m = train_model(real_sym, cfg)
        if m: 
            # Inisialisasi EMA distribution reference (FIX ISSUE 1)
            m['live_ref'] = pd.DataFrame(m['train_sample'])
            models_vault[real_sym] = m

last_candle_time = 0
update_correlation_matrix()
print("\n--- QUANT ENGINE V17.0 PARADIGM DEPLOYED ---")

while True:
    heartbeat() 
    
    if mt5.terminal_info() is None:
        mt5.shutdown(); time.sleep(5); mt5.initialize(); continue

    if account_kill_switch(): 
        time.sleep(600); continue

    if not models_vault: time.sleep(60); continue
    
    if datetime.now() - last_corr_update > timedelta(hours=2):
        update_correlation_matrix()
        last_corr_update = datetime.now()
        
    ref_sym = list(models_vault.keys())[0]
    df_ref = get_safe_rates(ref_sym, mt5.TIMEFRAME_M15, 1)
    
    if df_ref is not None and df_ref['time'].iloc[0] != last_candle_time:
        last_candle_time = df_ref['time'].iloc[0]
        update_loss_streak()

        if 2 <= datetime.now(timezone.utc).hour <= 21:
            for sym, m_data in list(models_vault.items()):
                
                if sym in disabled_symbols:
                    if datetime.now() < disabled_symbols[sym]: continue
                    else: del disabled_symbols[sym]

                if not check_model_health(sym): continue
                if not check_calibration_drift(sym): continue 

                recent_edge = get_live_edge(sym, datetime.now(timezone.utc))
                if recent_edge < -0.25: continue

                df_inf = get_safe_rates(sym, mt5.TIMEFRAME_M15, 600)
                if df_inf is None: continue
                
                last_bar_time = datetime.fromtimestamp(df_inf['time'].iloc[-1], tz=timezone.utc)
                if datetime.now(timezone.utc) - last_bar_time > timedelta(minutes=20): continue 
                
                df_inf['range_norm'] = (df_inf['high'] - df_inf['low']) / df_inf['ATR_14']
                df_inf['entropy'] = df_inf['range_norm'].rolling(30).apply(calc_entropy, raw=True)
                
                if df_inf['entropy'].iloc[-1] < df_inf['entropy'].rolling(200).quantile(0.25).iloc[-1]:
                    log_decision(sym, "NONE", 0, 0.0, m_data['threshold'], 0.0, 0.0, "SKIP_ENTROPY_GATE")
                    continue
                
                info = mt5.symbol_info(sym)
                df_inf.ta.ema(length=50, append=True)
                vol_now = df_inf['ATR_14'].iloc[-1]
                vol_mean = df_inf['ATR_14'].rolling(200).mean().iloc[-1]
                vol_ratio = vol_now / vol_mean
                
                if vol_ratio > 1.8 and datetime.now() - last_corr_update > timedelta(minutes=10):
                    update_correlation_matrix()
                    last_corr_update = datetime.now()
                
                df_inf['hl_range'] = df_inf['high'] - df_inf['low']
                spread_proxy = df_inf['hl_range'].rolling(5).median().iloc[-1]
                spread_norm = spread_proxy / vol_now
                tick = mt5.symbol_info_tick(sym)
                spread_points = (tick.ask - tick.bid) / info.point
                
                if sym not in spread_history: spread_history[sym] = []
                spread_history[sym].append(spread_points)
                if len(spread_history[sym]) > 100: spread_history[sym].pop(0)
                if spread_points > np.median(spread_history[sym]) * 2.2:
                    log_decision(sym, "NONE", 0, 0.0, m_data['threshold'], vol_ratio, spread_points, "SKIP_SPREAD_SHOCK")
                    continue

                df_inf['dist_ema_z'] = (df_inf['close'] - df_inf['EMA_50']) / df_inf['ATR_14']
                df_inf['vol_ratio'] = vol_ratio
                df_inf['roc_10'] = df_inf['close'].pct_change(10)
                df_inf['trend_strength'] = abs(df_inf['EMA_50'].pct_change(5)) / df_inf['ATR_14']
                df_inf['spread_norm'] = spread_norm
                df_inf['range_compress'] = (df_inf['high'].rolling(10).max() - df_inf['low'].rolling(10).min()) / df_inf['ATR_14']
                df_inf['clv'] = (df_inf['close'] - df_inf['low']) / (df_inf['high'] - df_inf['low'] + 1e-6)
                df_inf['vol_accel'] = df_inf['ATR_14'].pct_change(5)
                df_inf['kurtosis_50'] = df_inf['close'].pct_change().rolling(50).kurt()
                df_inf['skew_50'] = df_inf['close'].pct_change().rolling(50).skew()
                df_inf['realized_vol'] = df_inf['close'].pct_change().rolling(30).std()
                df_inf['vol_of_vol'] = df_inf['realized_vol'].rolling(30).std()
                df_inf['trend_persistence'] = abs(df_inf['EMA_50'].pct_change(10)) / (df_inf['realized_vol'] + 1e-6)
                df_inf['hour'] = pd.to_datetime(df_inf['time'], unit='s').dt.hour
                df_inf['session'] = np.select([df_inf['hour'] < 7, df_inf['hour'] < 13, df_inf['hour'] < 21], [0, 1, 2], default=3)
                
                candle_time = datetime.fromtimestamp(df_inf['time'].iloc[-1], tz=timezone.utc)
                df_inf['last_setup_outcome'] = get_live_outcome_feature(sym, candle_time)
                
                raw_live_edge = get_live_edge(sym, candle_time - timedelta(hours=6))
                df_inf['live_edge'] = np.tanh(raw_live_edge * 1.5)
                
                df_inf[m_data['feature_order']] = df_inf[m_data['feature_order']].shift(1)
                
                X_raw = df_inf[m_data['feature_order']].tail(1)
                if X_raw.isnull().values.any(): continue
                
                # FIX ISSUE 1: Update EMA Reference Distribution untuk PSI
                m_data['live_ref'] = 0.99 * m_data['live_ref'] + 0.01 * X_raw.values

                age_seconds = (datetime.now() - m_data['last_train']).total_seconds()
                needs_retrain = False
                
                if age_seconds > MODEL_MAX_AGE and model_trade_count.get(sym, 0) >= MIN_TRADES_FOR_RETRAIN:
                    needs_retrain = True
                elif age_seconds > 10800: 
                    live_window = df_inf[m_data['feature_order']].tail(200)
                    psi_features = ['vol_ratio', 'range_compress', 'trend_persistence', 'realized_vol']
                    psi_vals = [calculate_psi(m_data['live_ref'][f].values, live_window[f].dropna().values) 
                                for f in psi_features if f in m_data['feature_order'] and len(live_window[f].dropna()) > 0]
                    
                    if psi_vals and np.mean(psi_vals) > PSI_DRIFT_TRIGGER:
                        if model_trade_count.get(sym, 0) >= MIN_TRADES_FOR_RETRAIN:
                            needs_retrain = True
                
                if needs_retrain:
                    if mt5.positions_total() == 0:
                        recent_vol = df_inf['ATR_14'].tail(200)
                        vol_stability = recent_vol.std() / (recent_vol.mean() + 1e-6)
                        if vol_stability > 0.35:
                            print(f"[{datetime.now().strftime('%H:%M')}] [{sym}] Retrain Blocked: Unstable Regime")
                            m_data['last_train'] = datetime.now() - timedelta(seconds=MODEL_MAX_AGE - 3600) 
                            continue
                            
                        new_m = train_model(sym, m_data['config'])
                        if new_m and validate_model(new_m, models_vault[sym]):
                            new_m['live_ref'] = pd.DataFrame(new_m['train_sample'])
                            models_vault[sym] = new_m
                            model_trade_count[sym] = 0 
                            disabled_symbols[sym] = datetime.now() + timedelta(hours=2)
                    continue

                X_scaled = m_data['scaler'].transform(X_raw)
                
                regime_factor = np.clip(vol_ratio * df_inf['trend_strength'].iloc[-1], 0.5, 2.0)
                base_thresh = m_data['threshold']
                adaptive_threshold = base_thresh * (1 + 0.15 * (regime_factor - 1))
                loss_penalty = min(consecutive_losses * 0.015, 0.06)
                adaptive_threshold = np.clip(adaptive_threshold + loss_penalty, base_thresh, 0.92)
                
                manage_early_exits(sym, m_data, X_scaled, adaptive_threshold, vol_now)
                
                if vol_now < df_inf['ATR_14'].rolling(500).quantile(0.2).iloc[-1]: continue 
                
                probs = m_data['model'].predict_proba(X_scaled)[0]
                raw_pred = int(m_data['model'].classes_[np.argmax(probs)])
                conf = np.max(probs)
                action = "BUY" if raw_pred == 1 else "SELL" if raw_pred == -1 else "NONE"
                
                if conf >= adaptive_threshold and raw_pred != 0:
                    if vol_ratio < 0.6 or vol_ratio > 2.2:
                        log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, "SKIP_VOL_REGIME")
                        continue
                    if spread_points > m_data['config']['max_spread']:
                        log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, "SKIP_SPREAD")
                        continue
                    
                    execute_trade(sym, action, raw_pred, conf, m_data, adaptive_threshold, vol_ratio, spread_points, vol_now)
                else:
                    log_decision(sym, action, raw_pred, conf, adaptive_threshold, vol_ratio, spread_points, "SKIP_LOW_CONF")

    time.sleep(10)