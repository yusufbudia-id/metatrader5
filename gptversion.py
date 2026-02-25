import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from collections import deque
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15

BASE_RISK_PERCENT = 0.01
MODEL_MAX_AGE = 18 * 3600
MAGIC = 999

if not mt5.initialize():
    raise RuntimeError("MT5 not initialized")

# ==========================================
# DATABASE (AUTO MIGRATION SAFE)
# ==========================================
db_conn = sqlite3.connect("quant_production_log.db", check_same_thread=False)
db_cursor = db_conn.cursor()

db_cursor.execute("""
CREATE TABLE IF NOT EXISTS trade_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    symbol TEXT,
    action TEXT,
    conf REAL,
    threshold REAL,
    status TEXT,
    ticket INTEGER
)
""")
db_conn.commit()

def ensure_column(column, definition):
    cols = db_cursor.execute("PRAGMA table_info(trade_decisions)").fetchall()
    names = [c[1] for c in cols]
    if column not in names:
        print(f"[DB MIGRATION] adding {column}")
        db_cursor.execute(f"ALTER TABLE trade_decisions ADD COLUMN {column} {definition}")
        db_conn.commit()

ensure_column("expected_edge", "REAL DEFAULT 0.0")
ensure_column("realized_pnl", "REAL DEFAULT 0.0")

print("[DB] Schema OK")

# ==========================================
# DATA FETCH
# ==========================================
def get_rates(symbol, bars=5000):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df.ta.atr(length=14, append=True)
    df.ta.ema(length=50, append=True)
    return df.dropna()

# ==========================================
# FEATURE ENGINEERING
# ==========================================
FEATURES = [
    "dist_ema_z",
    "vol_ratio",
    "roc_10",
    "trend_strength"
]

def build_features(df):
    df["dist_ema_z"] = (df["close"] - df["EMA_50"]) / df["ATR_14"]
    df["vol_ratio"] = df["ATR_14"] / df["ATR_14"].rolling(200).mean()
    df["roc_10"] = df["close"].pct_change(10)
    df["trend_strength"] = abs(df["EMA_50"].pct_change(5)) / df["ATR_14"]
    df[FEATURES] = df[FEATURES].shift(1)
    return df.dropna()

def build_labels(df, horizon=12):
    future = df["close"].shift(-horizon)
    df["Target"] = np.where(future > df["close"], 1, 0)
    return df.dropna()

# ==========================================
# MODEL TRAINING
# ==========================================
def train_model(symbol):
    df = get_rates(symbol)
    if df is None:
        return None

    df = build_features(df)
    df = build_labels(df)

    if len(df) < 1000:
        return None

    train = df.iloc[:-300]
    val = df.iloc[-300:]

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train[FEATURES])

    base = GradientBoostingClassifier(n_estimators=120, max_depth=4)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, train["Target"])

    probs = model.predict_proba(scaler.transform(val[FEATURES]))
    threshold = np.quantile(np.max(probs, axis=1), 0.70)

    # IMPORTANT FIX → prevent starvation
    threshold = np.clip(threshold, 0.55, 0.80)

    print(f"[{symbol}] trained | threshold={threshold:.3f}")

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "last_train": datetime.now(),
        "live_buffer": deque(maxlen=1000)
    }

# ==========================================
# POSITION SIZE
# ==========================================
def calculate_lot(symbol, risk_percent, sl_points):
    info = mt5.symbol_info(symbol)
    acc = mt5.account_info()

    risk_money = acc.equity * risk_percent
    pip_value = info.trade_contract_size * info.point

    lot = risk_money / (sl_points * pip_value)
    lot = max(info.volume_min, min(lot, info.volume_max))

    step = info.volume_step
    return round(lot / step) * step

# ==========================================
# EXECUTION
# ==========================================
def execute_trade(symbol, action, confidence, model_data):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)

    atr = get_rates(symbol, 200)["ATR_14"].iloc[-1]
    sl_points = (2.0 * atr) / info.point

    lot = calculate_lot(symbol, BASE_RISK_PERCENT, sl_points)

    price = tick.ask if action == "BUY" else tick.bid

    sl = price - sl_points * info.point if action == "BUY" else price + sl_points * info.point
    tp = price + 3 * atr if action == "BUY" else price - 3 * atr

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 50,
        "magic": MAGIC,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    res = mt5.order_send(request)

    if res.retcode == mt5.TRADE_RETCODE_DONE:
        db_cursor.execute("""
        INSERT INTO trade_decisions(time,symbol,action,conf,threshold,status,ticket)
        VALUES(?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            symbol,
            action,
            confidence,
            model_data["threshold"],
            "EXECUTED",
            res.order
        ))
        db_conn.commit()

# ==========================================
# SYNC REALIZED PNL (FIXED)
# ==========================================
def sync_db_pnl():
    rows = db_cursor.execute("""
        SELECT ticket FROM trade_decisions
        WHERE realized_pnl=0.0 AND ticket IS NOT NULL
    """).fetchall()

    if not rows:
        return

    history = mt5.history_deals_get(datetime.now() - timedelta(days=5), datetime.now())
    if history is None:
        return

    pnl_map = {}
    for d in history:
        pnl_map.setdefault(d.position_id, 0)
        pnl_map[d.position_id] += d.profit

    for (ticket,) in rows:
        if ticket in pnl_map:
            db_cursor.execute("""
            UPDATE trade_decisions SET realized_pnl=?
            WHERE ticket=?
            """, (pnl_map[ticket], ticket))

    db_conn.commit()

# ==========================================
# INITIAL TRAIN
# ==========================================
models = {}
for sym in UNIVERSE:
    m = train_model(sym)
    if m:
        models[sym] = m

print("\n--- QUANT ENGINE RUNNING ---")

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    sync_db_pnl()

    for sym, m in models.items():

        # Retrain if expired
        if (datetime.now() - m["last_train"]).total_seconds() > MODEL_MAX_AGE:
            print(f"[{sym}] retraining...")
            new_m = train_model(sym)
            if new_m:
                models[sym] = new_m
            continue

        df = get_rates(sym, 300)
        df = build_features(df)

        X = m["scaler"].transform(df[FEATURES].tail(1))
        prob = m["model"].predict_proba(X)[0]

        conf = np.max(prob)

        if conf > m["threshold"]:
            action = "BUY" if np.argmax(prob) == 1 else "SELL"

            if not mt5.positions_get(symbol=sym):
                execute_trade(sym, action, conf, m)

    time.sleep(10)