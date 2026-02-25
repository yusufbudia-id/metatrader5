import MetaTrader5 as mt5
import pandas as pd

# 1. Inisialisasi Koneksi
print("Menghubungkan ke MT5...")
if not mt5.initialize():
    print("GAGAL: Tidak dapat terhubung ke MT5.")
    quit()

# 2. Pengaturan Parameter Data
symbol = "EURUSD"               # Anda bisa ganti dengan pair lain (misal: "XAUUSD" atau "GBPUSD")
timeframe = mt5.TIMEFRAME_H1    # Timeframe 1 Jam (Bisa diganti: mt5.TIMEFRAME_M15 untuk 15 menit)
num_candles = 1000              # Jumlah candle yang ingin diambil ke belakang

# Pastikan symbol/pair tersedia di Market Watch MT5 Anda
if not mt5.symbol_select(symbol, True):
    print(f"GAGAL: Symbol {symbol} tidak ditemukan. Pastikan ada di Market Watch.")
    mt5.shutdown()
    quit()

# 3. Mengambil Data dari MT5
print(f"Mengambil {num_candles} data candlestick {symbol}...")
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)

# 4. Mengubah Data Menjadi Tabel (Pandas DataFrame)
df = pd.DataFrame(rates)

# MT5 menyimpan waktu dalam format 'detik' (UNIX timestamp). 
# Kita ubah agar bisa dibaca manusia (Tahun-Bulan-Tanggal Jam:Menit:Detik)
df['time'] = pd.to_datetime(df['time'], unit='s')

# 5. Menampilkan Hasilnya
print("\n--- 5 Baris Pertama Data (Paling Lama) ---")
print(df.head())

print("\n--- 5 Baris Terakhir Data (Paling Baru / Saat Ini) ---")
print(df.tail())

# Menutup koneksi
mt5.shutdown()