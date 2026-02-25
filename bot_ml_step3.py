import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta  # Library baru untuk membuat indikator

# 1. AMBIL DATA DARI MT5 (Sama seperti Step 2)
if not mt5.initialize():
    print("GAGAL terhubung ke MT5")
    quit()

symbol = "EURUSD"
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
mt5.shutdown() # Matikan koneksi karena data sudah di tangan kita

# 2. FEATURE ENGINEERING (Membuat Fitur/Indikator untuk AI)
print("Sedang menghitung indikator teknikal...")

# Kita tambahkan 3 indikator populer sebagai "mata" untuk AI
df.ta.sma(length=10, append=True)  # Simple Moving Average (Cepat)
df.ta.sma(length=50, append=True)  # Simple Moving Average (Lambat)
df.ta.rsi(length=14, append=True)  # Relative Strength Index (Momentum)

# 3. MEMBUAT TARGET (Kunci Jawaban untuk AI)
# Logika: Jika harga 'close' jam DEPAN lebih besar dari jam INI, maka Target = 1 (Naik/Buy)
# Jika lebih kecil, maka Target = 0 (Turun/Sell)
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 4. MEMBERSIHKAN DATA
# Menghitung SMA 50 butuh 50 candle pertama, jadi 50 baris pertama akan kosong (NaN).
# AI benci data kosong, jadi kita hapus baris yang tidak lengkap.
df.dropna(inplace=True)

# 5. TAMPILKAN HASILNYA
print("\n--- Data Siap Latih (Buku Pelajaran AI) ---")
# Kita tampilkan kolom waktu, harga close, indikator, dan Target
print(df[['time', 'close', 'SMA_10', 'SMA_50', 'RSI_14', 'Target']].tail(10))