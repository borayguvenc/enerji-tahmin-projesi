import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Yolları ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from src.data_manager import DataManager
from src.model_manager import ModelManager, calculate_mape
from config import RAW_TARGET_COL

def run_simulation():
    print(" RECURSIVE FORECASTING SİMÜLASYONU BAŞLIYOR...\n")

    # 1. Veriyi ve Modeli Hazırla
    dm = DataManager()
    dm.load_and_preprocess()
    
    # Son 24 saati (1 Günü) Test için ayıralım
    # Simülasyonun net anlaşılması için kısa bir aralık seçiyoruz
    TEST_HORIZON = 24 
    
    # Tüm veri
    df = dm.data.copy()
    
    # Eğitim: Son 24 saat HARİÇ her şey
    train_df = df.iloc[:-TEST_HORIZON]
    # Test: Sadece son 24 saat
    test_df = df.iloc[-TEST_HORIZON:].copy()
    
    # Sadece modelin kabul ettiği veri tiplerini seç (Otomatik Filtreleme)
    feature_cols = df.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
    
    # Hedef değişkeni listeden çıkar
    if RAW_TARGET_COL in feature_cols:
        feature_cols.remove(RAW_TARGET_COL)
        

    
    X_train = train_df[feature_cols]
    y_train = train_df[RAW_TARGET_COL]
    X_test_static = test_df[feature_cols].copy() # Hileli (Gerçek verili) Test Seti
    y_test = test_df[RAW_TARGET_COL]

    # Modeli Eğit
    print("1️⃣ Model Eğitiliyor...")
    mm = ModelManager()
    mm.train_model(X_train, y_train, X_test_static, y_test)
    
    # ---------------------------------------------------------
    # SENARYO 1: STATIC (HİLELİ/MEVCUT DURUM) TAHMİN
    # ---------------------------------------------------------
    print("\n2️⃣ Static (Mevcut) Yöntemle Tahmin Yapılıyor...")
    # Burada Rolling_Mean sütunları zaten gerçek verilerle dolu
    preds_static = mm.model.predict(X_test_static)
    mape_static = calculate_mape(y_test, preds_static)
    print(f"   👉 Static MAPE: %{mape_static:.2f} (Bu skor yanıltıcı olabilir)")

    # ---------------------------------------------------------
    # SENARYO 2: RECURSIVE (GERÇEKÇİ) TAHMİN
    # ---------------------------------------------------------
    print("\n3️⃣ Recursive (Gerçek Hayat) Yöntemiyle Tahmin Yapılıyor...")
    
    # Başlangıç için X_test'in bir kopyasını alıyoruz ama içini güncelleyeceğiz
    X_test_recursive = X_test_static.copy()
    
    # Tahminleri saklayacağımız liste
    recursive_preds = []
    
    # Son bilinen gerçek değerleri al (Train setinin sonundan)
    # Rolling Mean hesaplamak için geçmişe ihtiyacımız var (Son 3 saat)
    last_known_values = list(y_train.iloc[-3:].values)
    
    # Adım adım (saat saat) ilerle
    for i in range(TEST_HORIZON):
        # Şu anki saatin satırını al (Tek satırlık DataFrame)
        current_row = X_test_recursive.iloc[[i]].copy()
        
        # --- KRİTİK NOKTA: ROLLING MEAN GÜNCELLEME ---
        # Elimizdeki 'last_known_values' listesindeki son 3 değerin ortalamasını al
        # Bu liste, döngü ilerledikçe bizim TAHMİNLERİMİZLE dolacak!
        new_rolling_mean = np.mean(last_known_values[-3:])
        
        # Eğer feature adın farklıysa burayı güncelle!
        if 'Rolling_Mean_3h' in current_row.columns:
            current_row['Rolling_Mean_3h'] = new_rolling_mean
        
        # 1. Tahmin Et
        pred = mm.model.predict(current_row)[0]
        recursive_preds.append(pred)
        
        # 2. Tahmini listeye ekle (Gelecek adımlar bunu kullanacak!)
        last_known_values.append(pred)
        
        # (Opsiyonel) Eğer Rolling_Mean_168h gibi başka lag'ler varsa onları da güncellemek gerekir
        # Ama şimdilik sadece 3h etkisine bakıyoruz.
    
    mape_recursive = calculate_mape(y_test, recursive_preds)
    print(f"   👉 Recursive MAPE: %{mape_recursive:.2f} (Gerçekçi Skor)")
    
    # ---------------------------------------------------------
    # SONUÇ VE GRAFİK
    # ---------------------------------------------------------
    diff = mape_recursive - mape_static
    print(f"\n⚠️ FARK: {diff:.2f} puan.")
    
    if diff > 2.0:
        print("❌ UYARI: Rolling Mean kullanımı ciddi hata yayılımına sebep oluyor!")
        print("   Öneri: Rolling_Mean_3h yerine Lag_24h tabanlı özellikler kullan.")
    else:
        print("✅ GÜVENLİ: Hata yayılımı tolere edilebilir seviyede.")

    # Grafik Çiz
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Gerçek (Actual)', color='black', linewidth=2)
    plt.plot(preds_static, label='Static (Hileli)', linestyle='--', color='blue')
    plt.plot(recursive_preds, label='Recursive (Gerçekçi)', linestyle='-', color='red', marker='o', markersize=4)
    plt.title(f"Recursive vs Static Tahmin (Fark: {diff:.2f} puan)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_simulation()