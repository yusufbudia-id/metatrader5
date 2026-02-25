import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import os
from datetime import datetime, timedelta, timezone # FIX 1: Tambah timezone
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. KONFIGURASI APEX EXECUTION
# ==========================================
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
magic_number = 777
max_risk_percent = 0.02
daily_loss_limit = 5000 
max_model_age_hours = 24
max_spread_pips = 2.0
entry_log_file = "quant_v15_entries.csv"
outcome_log_file = "quant_v15_outcomes.csv"

if not mt5.initialize(): quit()

open_pos = mt5.positions_get(symbol=symbol)
last_ticket = open_pos[0].ticket if open_pos else None
last_trade_time = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0][0]
loss_count = 0

def get_pip_unit(digits):
    return 0.01 if digits in [2, 3] else 0.0001

def get_lot_size(equity, sl_dist):
    info = mt5.symbol_info(symbol)
    if not info or sl_dist == 0: return 0.01
    risk_usd = equity * max_risk_percent
    lot = risk_usd / (sl_dist * (info.tick_value / info.tick_size))
    return round(max(min(lot, 10.0), 0.01), 2)

def record_last_trade_result(ticket):
    deals = mt5.history_deals_get(position=ticket)
    if not deals: return 0.0
    total_profit = sum(d.profit for d in deals)
    if total_profit > 0: return 2.0
    elif total_profit < 0: return -1.0
    return 0.0

# ==========================================
# 2. AI ENGINE 
# ==========================================
def create_directional_labels(df, window=20):
    labels = np.zeros(len(df))
    close, high, low = df['close'].values, df['high'].values, df['low'].values
    atr_vals = df['ATRr_14'].values 
    
    for i in range(len(df) - window):
        if np.isnan(atr_vals[i]): continue
        entry_p = close[i]
        sl_dist, tp_dist = 1.5 * atr_vals[i], 3.0 * atr_vals[i]
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
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> Training Apex Quant Engine...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 2500)
    df = pd.DataFrame(rates)
    
    df.ta.atr(length=14, append=True); df.ta.rsi(length=14, append=True); df.ta.adx(length=14, append=True)
    df['returns'] = df['close'].pct_change(); df['hour'] = pd.to_datetime(df['time'], unit='s').dt.hour
    
    recent_vol = df['ATRr_14'].tail(200).mean()
    long_vol = df['ATRr_14'].tail(1000).mean()
    if recent_vol / long_vol > 2.0:
        print(">>> WARNING: Volatility Drift Detected. Training Aborted.")
        return None, None, None, None

    feature_cols = [c for c in df.columns if any(x in c for x in ['RSI', 'ADX', 'returns', 'hour', 'ATRr'])]
    
    df[feature_cols] = df[feature_cols].shift(1)
    df['Target'] = create_directional_labels(df)
    df.dropna(inplace=True)
    
    X, y = df[feature_cols], df['Target']
    split = int(len(df) * 0.8)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=30, random_state=42)
    
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    model.fit(X_train, y_train)
    
    # FIX 2: Hitung Threshold hanya dari sinyal AKTIF (Mencegah Zero-Class Domination)
    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)
    active_mask = train_preds != 0
    
    if np.sum(active_mask) > 10:
        active_probs = np.max(train_probs[active_mask], axis=1)
        threshold = np.quantile(active_probs, 0.60) # Top 40% dari tebakan aktif saja
    else:
        threshold = 0.55 # Fallback jika model sangat pelit entry
    
    pred_out = model.predict(X_test)
    real = y_test.values
    pnl = np.where(pred_out == real, np.where(real != 0, 2.0, 0.0), np.where(pred_out != 0, -1.0, 0.0))  
    
    edge = pnl.mean()
    print(f">>> Expected R (Edge): {edge:.4f}R | Active Threshold: {threshold:.2f}")
    return model, feature_cols, threshold, edge

# ==========================================
# 3. LIVE DECISION & PROTECTED EXECUTION
# ==========================================
if not os.path.exists(entry_log_file):
    with open(entry_log_file, "w") as f: f.write("time,ticket,action,lot,conf,atr,edge_R\n")
if not os.path.exists(outcome_log_file):
    with open(outcome_log_file, "w") as f: f.write("time,ticket,realized_R\n")

model_ai, features, threshold, current_edge = latih_ai_pro()
last_train = datetime.now()

while True:
    # FIX 1: Gunakan timezone-aware UTC datetime
    hour_utc = datetime.now(timezone.utc).hour
    if hour_utc < 6 or hour_utc > 20:
        time.sleep(60); continue

    if model_ai is None:
        print("Model unavailable due to drift. Pausing trading for 15 mins...")
        time.sleep(900)
        model_ai, features, threshold, current_edge = latih_ai_pro()
        last_train = datetime.now()
        continue

    acc_info = mt5.account_info()
    if acc_info.equity < (acc_info.balance - daily_loss_limit):
        print("!!! HARD KILL-SWITCH ACTIVATED !!!"); break

    if loss_count >= 5:
        print("!!! Loss Cluster Detected. Cooling down for 30 minutes. !!!")
        time.sleep(1800)
        loss_count = 0
        continue 

    if (datetime.now() - last_train).total_seconds() > max_model_age_hours * 3600:
        print("!!! MODEL EXPIRED: Trading Paused !!!")
        new_m, new_f, new_t, new_e = latih_ai_pro()
        if new_m: model_ai, features, threshold, current_edge, last_train = new_m, new_f, new_t, new_e, datetime.now()
        time.sleep(60); continue

    if (datetime.now() - last_train).total_seconds() > 14400 or loss_count >= 3:
        new_m, new_f, new_t, new_e = latih_ai_pro()
        if new_m: 
            model_ai, features, threshold, current_edge = new_m, new_f, new_t, new_e
            last_train = datetime.now()

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
        df_l.ta.atr(length=14, append=True); df_l.ta.rsi(length=14, append=True); df_l.ta.adx(length=14, append=True)
        df_l['hour'] = pd.to_datetime(df_l['time'], unit='s').dt.hour; df_l['returns'] = df_l['close'].pct_change()
        
        atr_now = df_l['ATRr_14'].iloc[-1]
        long_term_atr = df_l['ATRr_14'].rolling(500).mean().iloc[-1]
        
        if pd.notna(long_term_atr) and atr_now < long_term_atr * 0.5:
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
            sl_dist, tp_dist = 1.5 * atr_now, 3.0 * atr_now
            action = "BUY" if pred == 1 else "SELL"
            price = tick.ask if action == "BUY" else tick.bid
            lot_size = get_lot_size(acc_info.equity, sl_dist)
            
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
                    f.write(f"{datetime.now()},{last_ticket},{action},{res.volume},{conf:.2f},{atr_now:.5f},{current_edge:.4f}\n")
                print(f"[{datetime.now().strftime('%H:%M')}] {action} Executed (Lot: {lot_size}). Conf: {conf:.2f}")

    time.sleep(30)