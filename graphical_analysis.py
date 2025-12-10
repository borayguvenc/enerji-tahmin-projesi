import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# --- YOL AYARLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_manager import DataManager
from src.model_manager import ModelManager
from config import RAW_TARGET_COL

def analyze_specific_period(start_date, end_date):
    print(f"\n {start_date} ile {end_date} arasındaki veriler inceleniyor...")

    # 1. Veriyi Yükle
    dm = DataManager()
    df = dm.load_and_preprocess()

    # 2. İlgili Tarih Aralığını Kes (Filter)
    # Pandas string ile tarih aralığı seçmemize izin verir
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_period = df.loc[mask].copy()

    if df_period.empty:
        print("HATA: Seçilen tarih aralığında veri bulunamadı!")
        return

    # 3. Modeli Yükle ve Tahmin Yap
    mm = ModelManager()
    mm.load_model() # Kaydedilmiş en son modeli yükler
    
    if mm.model is None:
        print("Model yüklenemedi. Önce eğitimi çalıştırın.")
        return

    # Özellikleri ayır (DataManager'daki mantığın aynısı)
 
    feature_cols = [c for c in df_period.columns if c not in [RAW_TARGET_COL, 'Tarih']]
    X_period = df_period[feature_cols]
    y_true = df_period[RAW_TARGET_COL]

    # Tahmin Üret
    preds = mm.model.predict(X_period)
    
    # 4. Grafiği Çiz
    plt.figure(figsize=(15, 7))
    
    # Gerçek Tüketim (Mavi)
    plt.plot(df_period.index, y_true, label='Gerçek Tüketim (Real)', color='blue', linewidth=2, alpha=0.7)
    
    # Tahmin (Kırmızı - Kesikli)
    plt.plot(df_period.index, preds, label='Model Tahmini (Pred)', color='red', linestyle='--', linewidth=2)

    # --- Görselliği İyileştirme ---
    plt.title(f'Hata Analizi: {start_date} / {end_date}', fontsize=14)
    plt.ylabel('Elektrik Tüketimi (MWh)', fontsize=12)
    plt.xlabel('Tarih', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # Kurban Bayramı Dönemini İşaretle (Tahmini: 6-9 Haziran 2025)
    # Grafikte o bölgeyi gri boyayalım ki gözüne çarpsın
    plt.axvspan(pd.Timestamp('2025-06-05'), pd.Timestamp('2025-06-10'), color='gray', alpha=0.2, label='Kurban Bayramı (Tahmini)')

    plt.tight_layout()
    plt.show()

    # 5. Sayısal Farkı da Yazdır (Toplam Hata)
    total_real = np.sum(y_true)
    total_pred = np.sum(preds)
    diff_perc = ((total_pred - total_real) / total_real) * 100
    
    print(f"\n--- İSTATİSTİKLER ---")
    print(f"Toplam Gerçek Tüketim: {total_real:,.0f}")
    print(f"Toplam Tahmin        : {total_pred:,.0f}")
    print(f"Fark (Bias)          : %{diff_perc:.2f}")
    
    if diff_perc > 0:
        print("👉 SONUÇ: Model genel olarak FAZLA tahmin yapmış (Over-prediction).")
    else:
        print("👉 SONUÇ: Model genel olarak EKSİK tahmin yapmış (Under-prediction).")

# --- ÇALIŞTIR ---
if __name__ == "__main__":
    # Senin sorunlu tarih aralığın
    analyze_specific_period('2025-06-02', '2025-07-02')