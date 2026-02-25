import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
from datetime import datetime, timedelta, timezone 
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. KONFIGURASI HEDGE FUND CORE
# ==========================================
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
magic_number = 777
max_risk_percent = 0.02
daily_loss_limit = 5000 
max_model_age_hours = 24
max_spread_pips = 2.0
entry_log_file = "quant_v17_entries.csv"
outcome_log_file = "quant_v17_outcomes.csv"

if not mt5.initialize(): quit()

open_pos = mt5.positions_get(symbol=symbol)
last_ticket = open_pos[0].ticket if open_pos else None
last_trade_time = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0][0]
loss_count = 0

def get_pip_unit(digits):
    return 0.01 if digits in [2, 3] else 0.0001

def get_lot_size(equity, sl_dist, confidence, threshold):
    info = mt5.symbol_info(symbol)
    if not info or sl_dist == 0: return 0.01
    
    if threshold >= 1.0: threshold = 0.99
    edge_factor = (confidence - threshold) / (1.0 - threshold)
    edge_factor = max(min(edge_factor, 1.0), 0.2)  
    
    risk_usd = equity * max_risk_percent * edge_factor
    lot = risk_usd / (sl_dist * (info.tick_value / info.tick_size))
    return round(max(min(lot, 10.0), 0.01), 2)

def record_last_trade_result(ticket):
    deals = mt5.history_deals_get(position=ticket)
    if not deals: return 0.0
    total_profit = sum(d.profit for d in deals)
    if total_profit > 0: return 1.5 
    elif total_profit < 0: return -1.0
    return 0.0

# ==========================================
# 2. AI ENGINE (REGIME AWARENESS)
# ==========================================
def create_directional_labels(df, window=40): 
    labels = np.zeros(len(df))
    close, high, low = df['close'].values, df['high'].values, df['low'].values
    atr_vals = df['ATRr_14'].values 
    
    for i in range(len(df) - window):
        if np.isnan(atr_vals[i]): continue
        entry_p = close[i]
        sl_dist, tp_dist = 1.5 * atr_vals[i], 2.25 * atr_vals[i] 
        buy_tp, buy_sl = entry_p + tp_dist, entry_p - sl_dist
        sell_tp, sell_sl = entry_p - tp_dist, entry_p + sl_dist
        
        for j in range(1, window + 1):
            curr_h, curr_l = high[i+j], low[i+j]
            h_btp, h_bsl = curr_h >= buy_tp, curr_l <= buy_sl
            h_stp, h_ssl = curr_l <= sell_tp, curr_h >= sell_sl
            
            if h_btp and not h_stp: labels[i] = 1; break
            elif h_stp and not h_btp: labels[i] = -1; break
            elif h_bsl or h_ssl: break
    return labels

def latih_ai_pro():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> Training Hedge Fund Quant Core...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 3000)
    df = pd.DataFrame(rates)
    
    df.ta.atr(length=14, append=True); df.ta.ema(length=50, append=True)
    df['dist_ema_z'] = (df['close'] - df['EMA_50']) / df['ATRr_14']
    df['vol_ratio'] = df['ATRr_14'] / df['ATRr_14'].rolling(100).mean()
    df['roc_10'] = df['close'].pct_change(10)
    df['returns'] = df['close'].pct_change(); df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    df['trend_strength'] = abs(df['EMA_50'].pct_change(5)) / df['ATRr_14']
    
    recent_vol = df['ATRr_14'].tail(200).mean()
    long_vol = df['ATRr_14'].tail(1000).mean()
    if recent_vol / long_vol > 2.0:
        print(">>> WARNING: Volatility Drift Detected. Training Aborted.")
        return None, None, None, None

    feature_cols = [c for c in df.columns if any(x in c for x in ['dist_ema_z', 'vol_ratio', 'roc_10', 'returns', 'hour', 'trend_strength'])]
    
    df[feature_cols] = df[feature_cols].shift(1)
    df['Target'] = create_directional_labels(df)
    df.dropna(inplace=True)
    
    X, y = df[feature_cols], df['Target']
    split = int(len(df) * 0.8)
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=25, 
        random_state=42, class_weight='balanced'
    )
    
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)
    active_mask = train_preds != 0
    
    if np.sum(active_mask) > 10:
        active_probs = np.max(train_probs[active_mask], axis=1)
        threshold = np.quantile(active_probs, 0.65)
    else:
        threshold = 0.55 
    
    pred_out = model.predict(X_test)
    real = y_test.values
    pnl = np.where(pred_out == real, np.where(real != 0, 1.5, 0.0), np.where(pred_out != 0, -1.0, 0.0))  
    
    edge = pnl.mean()
    print(f">>> Expected R (Edge): {edge:.4f}R | Active Threshold: {threshold:.2f}")
    if edge < 0: print("    [!] WARNING: Historical edge is still negative. Trade with absolute caution.")
    return model, feature_cols, threshold, edge

# ==========================================
# 3. LIVE DECISION & PROTECTED EXECUTION
# ==========================================
if not os.path.exists(entry_log_file):
    with open(entry_log_file, "w") as f: f.write("time,ticket,action,lot,conf,atr,edge_R,trend_str\n")
if not os.path.exists(outcome_log_file):
    with open(outcome_log_file, "w") as f: f.write("time,ticket,realized_R\n")

model_ai, features, threshold, current_edge = latih_ai_pro()
last_train = datetime.now()

while True:
    hour_utc = datetime.now(timezone.utc).hour
    if hour_utc < 6 or hour_utc > 20:
        time.sleep(60); continue

    if model_ai is None:
        print("Model unavailable due to drift. Pausing...")
        time.sleep(900)
        model_ai, features, threshold, current_edge = latih_ai_pro()
        last_train = datetime.now(); continue

    acc_info = mt5.account_info()
    if acc_info.equity < (acc_info.balance - daily_loss_limit):
        print("!!! HARD KILL-SWITCH ACTIVATED !!!"); break

    if loss_count >= 5:
        print("!!! Loss Cluster Detected. Cooling down for 30 minutes. !!!")
        time.sleep(1800)
        loss_count = 0; continue 

    if (datetime.now() - last_train).total_seconds() > max_model_age_hours * 3600:
        print("!!! MODEL EXPIRED: Trading Paused !!!")
        new_m, new_f, new_t, new_e = latih_ai_pro()
        if new_m: model_ai, features, threshold, current_edge, last_train = new_m, new_f, new_t, new_e, datetime.now()
        time.sleep(60); continue

    # ==========================================
    # HOTFIX V17.1: ANTI-SPAM & NEGATIVE EDGE LOCK
    # ==========================================
    time_since_train = (datetime.now() - last_train).total_seconds()
    
    # Hanya trigger retrain jika sudah minimal 1 jam berlalu (mencegah loop panik)
    if loss_count >= 3 or (current_edge < -0.05 and time_since_train > 3600):
        print(f"\n!!! Event-Triggered Retrain (Losses: {loss_count}, Edge: {current_edge:.4f}) !!!")
        new_m, new_f, new_t, new_e = latih_ai_pro()
        if new_m: 
            model_ai, features, threshold, current_edge = new_m, new_f, new_t, new_e
            last_train = datetime.now()
            loss_count = 0 

    # JIKA EDGE NEGATIF = ROBOT TIDUR. DILARANG TRADING.
    if current_edge < 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] STRATEGY OFFLINE: Edge Negatif ({current_edge:.4f}R). Market tidak kondusif. Sleep 1 jam...")
        time.sleep(3600) # Tidur 1 Jam
        continue # Lewati logika trading di bawah ini
    # ==========================================

    pos = mt5.positions_get(symbol=symbol)
    curr_bar_time = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0][0]

    if not pos:
        if last_ticket is not None:
            realized_R = record_last_trade_result(last_ticket)
            if realized_R < 0: loss_count += 1
            else: loss_count = 0
            
            with open(outcome_log_file, "a") as f:
                f.write(f"{datetime.now()},{last_ticket},{realized_R}\n")
            print(f"[{datetime.now().strftime('%H:%M')}] Trade Closed | Realized R: {realized_R}R")
            last_ticket = None

        if curr_bar_time - last_trade_time < (15 * 60 * 3):
            time.sleep(10); continue

        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        spread = (tick.ask - tick.bid) / get_pip_unit(info.digits)
        if spread > max_spread_pips:
            time.sleep(10); continue

        raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, 600)
        df_l = pd.DataFrame(raw)
        df_l.ta.atr(length=14, append=True); df_l.ta.ema(length=50, append=True)
        df_l['dist_ema_z'] = (df_l['close'] - df_l['EMA_50']) / df_l['ATRr_14']
        df_l['vol_ratio'] = df_l['ATRr_14'] / df_l['ATRr_14'].rolling(100).mean()
        df_l['roc_10'] = df_l['close'].pct_change(10)
        df_l['hour'] = pd.to_datetime(df_l['time'], unit='s').dt.hour; df_l['returns'] = df_l['close'].pct_change()
        
        df_l['trend_strength'] = abs(df_l['EMA_50'].pct_change(5)) / df_l['ATRr_14']
        
        atr_now = df_l['ATRr_14'].iloc[-1]
        long_term_atr = df_l['ATRr_14'].rolling(500).mean().iloc[-1]
        
        if pd.notna(long_term_atr) and atr_now < long_term_atr * 0.5:
            time.sleep(10); continue

        trend_strength_now = df_l['trend_strength'].iloc[-1]
        if pd.notna(trend_strength_now) and trend_strength_now < 0.15:
            time.sleep(10); continue

        df_l[features] = df_l[features].shift(1)
        X_now = df_l[features].tail(1)
        
        if X_now.isnull().values.any(): 
            time.sleep(5); continue 
        
        probs = model_ai.predict_proba(X_now)[0]
        conf, pred = np.max(probs), model_ai.classes_[np.argmax(probs)]

        vol_ratio = atr_now / df_l['ATRr_14'].rolling(200).mean().iloc[-1]
        adaptive_threshold = threshold * np.clip(vol_ratio, 0.9, 1.1)

        if conf >= adaptive_threshold and pred != 0:
            sl_dist, tp_dist = 1.5 * atr_now, 2.25 * atr_now 
            action = "BUY" if pred == 1 else "SELL"
            price = tick.ask if action == "BUY" else tick.bid
            
            lot_size = get_lot_size(acc_info.equity, sl_dist, conf, adaptive_threshold)
            
            res = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price, "sl": price - sl_dist if action == "BUY" else price + sl_dist,
                "tp": price + tp_dist if action == "BUY" else price - tp_dist,
                "magic": magic_number, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            })
            
            if res.retcode == mt5.TRADE_RETCODE_DONE and res.volume > 0:
                last_trade_time = curr_bar_time
                last_ticket = res.order 
                
                with open(entry_log_file, "a") as f:
                    f.write(f"{datetime.now()},{last_ticket},{action},{res.volume},{conf:.2f},{atr_now:.5f},{current_edge:.4f},{trend_strength_now:.4f}\n")
                print(f"[{datetime.now().strftime('%H:%M')}] {action} Executed (Lot: {lot_size}). Conf: {conf:.2f} | TrendStr: {trend_strength_now:.2f}")

    time.sleep(30)
