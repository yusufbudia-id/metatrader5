import MetaTrader5 as mt5

# 1. Hubungkan ke MT5
if not mt5.initialize():
    print("GAGAL: Tidak terhubung ke MT5")
    quit()

# 2. Pengaturan Order
symbol = "EURUSD"
lot_size = 0.01      
magic_number = 999   

# Pastikan symbol tersedia dan ambil informasinya
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} tidak ditemukan!")
    mt5.shutdown()
    quit()

if not symbol_info.visible:
    mt5.symbol_select(symbol, True)

# --- INI ADALAH KUNCI PERBAIKANNYA ---
# Kita gunakan angka mentah untuk menghindari error versi library
# 1 = FOK, 2 = IOC, 3 = Mendukung Keduanya
mode = symbol_info.filling_mode
if mode == 1 or mode == 3:
    filling_type = mt5.ORDER_FILLING_FOK
elif mode == 2:
    filling_type = mt5.ORDER_FILLING_IOC
else:
    filling_type = mt5.ORDER_FILLING_RETURN

# 3. Fungsi untuk mengirim Order Buy/Sell
def kirim_order(action, symbol, lot):
    tick = mt5.symbol_info_tick(symbol)
    if action == "BUY":
        tipe_order = mt5.ORDER_TYPE_BUY
        harga = tick.ask
    elif action == "SELL":
        tipe_order = mt5.ORDER_TYPE_SELL
        harga = tick.bid
    else:
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": tipe_order,
        "price": harga,
        "deviation": 20,   
        "magic": magic_number,
        "comment": "Order ML Python",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type, # Menggunakan aturan yang disetujui broker
    }

    hasil = mt5.order_send(request)
    return hasil

# ==========================================
# 4. SIMULASI KEPUTUSAN AI
# ==========================================
print("--- Simulasi Eksekusi Trading ---")
prediksi_ai = 1  # Simulasi AI menebak NAIK

if prediksi_ai == 1:
    print("AI Memprediksi NAIK! Mengirim perintah BUY ke MT5...")
    hasil_order = kirim_order("BUY", symbol, lot_size)
else:
    print("AI Memprediksi TURUN! Mengirim perintah SELL ke MT5...")
    hasil_order = kirim_order("SELL", symbol, lot_size)

# Cek apakah order sukses masuk ke pasar
if hasil_order is None:
    print("GAGAL: Request tidak terkirim sama sekali.")
elif hasil_order.retcode == mt5.TRADE_RETCODE_DONE:
    print("SUKSES: Posisi berhasil dibuka di MT5!")
else:
    print(f"GAGAL: Terjadi kesalahan. Kode Error: {hasil_order.retcode}")

mt5.shutdown()