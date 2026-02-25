import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- BAGIAN 1 & 2: Ambil Data & Buat Indikator (Sama seperti sebelumnya) ---
mt5.initialize()
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 1000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
mt5.shutdown()

df.ta.sma(length=10, append=True)
df.ta.sma(length=50, append=True)
df.ta.rsi(length=14, append=True)
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# --- BAGIAN 3: PROSES MACHINE LEARNING ---
print("Memulai proses pelatihan AI...")

# 1. Tentukan "Buku Pelajaran" (Fitur/X) dan "Kunci Jawaban" (Target/y)
X = df[['SMA_10', 'SMA_50', 'RSI_14']]  # AI hanya akan melihat 3 indikator ini
y = df['Target']                        # AI akan menebak ini (1 = Naik, 0 = Turun)

# 2. Bagi Data (80% untuk Belajar, 20% untuk Ujian)
# CATATAN PENTING: shuffle=False wajib di trading agar AI tidak mengintip masa depan!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Jumlah data untuk AI belajar : {len(X_train)} candle")
print(f"Jumlah data untuk AI ujian   : {len(X_test)} candle")

# 3. Panggil Model AI (Random Forest dengan 100 "trader virtual")
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Latih AI-nya! (Proses belajar)
model.fit(X_train, y_train)
print("AI selesai belajar!")

# 5. Uji AI-nya pada data yang belum pernah ia lihat (Data Ujian / Test)
prediksi_ai = model.predict(X_test)

# 6. Hitung Nilai Ujian AI (Akurasi)
akurasi = accuracy_score(y_test, prediksi_ai)
print(f"\n---> HASIL UJIAN AI: Akurasi Tebakan = {akurasi * 100:.2f}% <---")

# Tampilkan beberapa contoh tebakan vs kenyataan di baris terakhir
hasil_df = pd.DataFrame({
    'Tebakan AI': prediksi_ai[-5:], 
    'Kenyataan (Target)': y_test.values[-5:]
})
print("\nContoh 5 Tebakan Terakhir AI vs Kenyataan (1=Naik, 0=Turun):")
print(hasil_df)