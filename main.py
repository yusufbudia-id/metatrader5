import MetaTrader5 as mt5
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# 1. Inisialisasi Koneksi ke MT5
if not mt5.initialize():
    print("Gagal terhubung ke MT5")
    quit()

symbol = "EURUSD"
volume = 0.1

# 2. Fungsi Ambil Data & Buat Fitur (Feature Engineering)
def get_data_and_features():
    # Ambil 1000 candle terakhir
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 1000)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Buat fitur sederhana (Misal: Perbedaan harga Close ke Open)
    df['Price_Change'] = df['close'] - df['open']
    
    # Tentukan Target: 1 jika harga naik, 0 jika turun
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna()

# 3. Latih Model AI
print("Melatih Model AI...")
data = get_data_and_features()
X = data[['Price_Change']] # Fitur input
y = data['Target']         # Jawaban (Naik/Turun)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
print("Model siap!")

# 4. Live Trading Loop
while True:
    # Ambil data candle paling baru
    latest_data = get_data_and_features().tail(1)
    current_features = latest_data[['Price_Change']]
    
    # AI Memprediksi
    prediction = model.predict(current_features)
    
    # Eksekusi Order
    if prediction[0] == 1:
        print("AI Memprediksi NAIK -> Eksekusi BUY")
        # Kode eksekusi Buy ke MT5 ditaruh di sini
    else:
        print("AI Memprediksi TURUN -> Eksekusi SELL")
        # Kode eksekusi Sell ke MT5 ditaruh di sini
        
    time.sleep(900) # Tunggu 15 menit untuk candle selanjutnya