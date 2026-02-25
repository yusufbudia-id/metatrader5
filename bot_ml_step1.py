import MetaTrader5 as mt5
import pandas as pd

print("Memulai proses koneksi ke MetaTrader 5...")

# 1. Inisialisasi koneksi ke terminal MT5
if not mt5.initialize():
    print("GAGAL: Tidak dapat terhubung ke MT5.")
    print("Error code:", mt5.last_error())
    # Pastikan aplikasi MT5 sedang terbuka di background
    quit()

print("SUKSES: Python berhasil terhubung ke terminal MT5!")

# 2. Mengambil informasi akun yang sedang login di MT5
account_info = mt5.account_info()

if account_info != None:
    print("\n--- Informasi Akun Trading ---")
    print(f"Nomor Akun : {account_info.login}")
    print(f"Server     : {account_info.server}")
    print(f"Saldo      : $ {account_info.balance}")
    print(f"Equity     : $ {account_info.equity}")
else:
    print("\nGagal mengambil data akun. Pastikan Anda sudah login ke akun di MT5.")

# 3. Menutup koneksi setelah selesai (Praktik yang baik untuk testing)
mt5.shutdown()
print("\nKoneksi ditutup dengan aman.")