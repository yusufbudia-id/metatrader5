import sqlite3
import pandas as pd

# Terhubung ke database micro
conn = sqlite3.connect("quant_production_log_micro.db")

# Mengambil 10 keputusan terakhir
query = "SELECT time, symbol, action, conf, threshold, spread, status FROM trade_decisions ORDER BY id DESC LIMIT 10"
df = pd.read_sql_query(query, conn)

if df.empty:
    print("Belum ada log di database. Tunggu sampai pergantian candle 15 menit berikutnya!")
else:
    print("\n--- 10 KEPUTUSAN TERAKHIR BOT ---")
    print(df.to_string(index=False))

conn.close()