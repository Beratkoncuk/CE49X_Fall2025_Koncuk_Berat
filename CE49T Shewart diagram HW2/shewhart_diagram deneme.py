# shewhart_from_excel.py
# Kullanım: aynı klasörde bulunan 'coil_resistance_ordered.xlsx' dosyasını okuyup
# Shewhart kontrol grafiğini oluşturur ve 'shewhart_from_excel.png' olarak kaydeder.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Ayarlar ---
EXCEL_PATH = r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\CE 49T\coil_resistance_ordered.xlsx"   # Excel dosyası adı (aynı klasörde olmalı)
SHEET_NAME = 0                                 # Varsayılan ilk sheet
OUTPUT_PNG = "shewhart_from_excel.png"
FIGSIZE = (12, 6)
MARKER_SIZE = 6

# --- 1) Veri okuma ---
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"Excel dosyası bulunamadı: {EXCEL_PATH}\n"
                            "Dosyanın script ile aynı klasörde olduğundan emin ol.")

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

# Beklenen sütun isimleri: 'Coil No' ve 'Resistance (ohm)'
# Eğer farklıysa ilk iki sütunu kullan:
if df.shape[1] >= 2:
    coil_col = df.columns[0]
    res_col = df.columns[1]
else:
    raise ValueError("Excel dosyasında en az iki sütun (Coil No ve Resistance) bekleniyor.")

# Temiz veri
df = df[[coil_col, res_col]].dropna().reset_index(drop=True)
df.columns = ["CoilNo", "Resistance"]

# Eğer CoilNo sıralı değilse sıralı hale getir (opsiyonel)
df = df.sort_values("CoilNo").reset_index(drop=True)

# --- 2) İstatistik hesapları ---
n = len(df)
mean = df["Resistance"].mean()
std_sample = df["Resistance"].std(ddof=1)  # örnek standart sapma (ödevde ddof=1 uygundur)

ucl1 = mean + 1*std_sample
lcl1 = mean - 1*std_sample
ucl2 = mean + 2*std_sample
lcl2 = mean - 2*std_sample
ucl3 = mean + 3*std_sample
lcl3 = mean - 3*std_sample

# --- 3) Kontrol kurallarına göre anormallikleri tespit et (basit) ---
out_of_3sigma = df[(df["Resistance"] > ucl3) | (df["Resistance"] < lcl3)]
out_of_2sigma = df[(df["Resistance"] > ucl2) | (df["Resistance"] < lcl2)]
out_of_1sigma = df[(df["Resistance"] > ucl1) | (df["Resistance"] < lcl1)]

# --- 4) Grafik çizimi ---
plt.figure(figsize=FIGSIZE)
x = df["CoilNo"].to_numpy()
y = df["Resistance"].to_numpy()

# noktalar
plt.plot(x, y, marker='o', linestyle='-', markersize=MARKER_SIZE, label="Ölçümler")

# ortalama ve kontrol limitleri
plt.axhline(mean, color='black', linestyle='-', linewidth=1.2, label=f"Ortalama = {mean:.4f}")
plt.axhline(ucl1, color='tab:orange', linestyle='--', linewidth=1, label=f"UCL/LCL ±1σ = {ucl1:.4f}/{lcl1:.4f}")
plt.axhline(lcl1, color='tab:orange', linestyle='--', linewidth=1)
plt.axhline(ucl2, color='tab:green', linestyle='--', linewidth=1, label=f"±2σ = {ucl2:.4f}/{lcl2:.4f}")
plt.axhline(lcl2, color='tab:green', linestyle='--', linewidth=1)
plt.axhline(ucl3, color='tab:red', linestyle='--', linewidth=1, label=f"±3σ = {ucl3:.4f}/{lcl3:.4f}")
plt.axhline(lcl3, color='tab:red', linestyle='--', linewidth=1)

# 3σ dışındaki noktaları kırmızıya boya ve işaretle
if not out_of_3sigma.empty:
    plt.scatter(out_of_3sigma["CoilNo"], out_of_3sigma["Resistance"], color='red', s=80, zorder=5, label="3σ dışı noktalar")
    for _, row in out_of_3sigma.iterrows():
        plt.text(row["CoilNo"], row["Resistance"], f'  {int(row["CoilNo"])}', va='bottom', fontsize=9)

# 2σ-3σ arasındaki noktaları (varsa) farklı göster (örnek görsellik)
between_2_3 = df[((df["Resistance"] > ucl2) & (df["Resistance"] <= ucl3)) | 
                 ((df["Resistance"] < lcl2) & (df["Resistance"] >= lcl3))]
if not between_2_3.empty:
    plt.scatter(between_2_3["CoilNo"], between_2_3["Resistance"], marker='D', s=70, zorder=5, label="2σ-3σ arası")

# görsellik ve etiketleme
plt.title("Shewhart Kontrol Grafiği — Coil Resistance")
plt.xlabel("Coil Number")
plt.ylabel("Resistance (ohm)")
plt.grid(axis='y', linestyle=':', linewidth=0.6)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()

# Kaydet ve göster
plt.savefig(OUTPUT_PNG, dpi=300)
print(f"Grafik kaydedildi: {OUTPUT_PNG}")
plt.show()

# --- 5) Konsola kısa özet yazdır ---
print("\n--- Özet ---")
print(f"Örnek sayısı (n): {n}")
print(f"Ortalama (x̄): {mean:.6f}")
print(f"Örnek standart sapma (s): {std_sample:.6f}")
print(f"UCL/LCL 1σ: {ucl1:.6f} / {lcl1:.6f}")
print(f"UCL/LCL 2σ: {ucl2:.6f} / {lcl2:.6f}")
print(f"UCL/LCL 3σ: {ucl3:.6f} / {lcl3:.6f}")

print("\n3σ dışındaki noktalar (varsa):")
if out_of_3sigma.empty:
    print("Yok")
else:
    print(out_of_3sigma.to_string(index=False))

# Opsiyonel: CSV'ye limitleri yazdırmak istersen:
limits_df = pd.DataFrame({
    "Metric": ["mean", "std_sample", "UCL1", "LCL1", "UCL2", "LCL2", "UCL3", "LCL3"],
    "Value": [mean, std_sample, ucl1, lcl1, ucl2, lcl2, ucl3, lcl3]
})
limits_df.to_csv("shewhart_limits_summary.csv", index=False)
print("\nLimit özeti 'shewhart_limits_summary.csv' olarak kaydedildi.")
