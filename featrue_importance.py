import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Proje yollarını ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.data_manager import DataManager
from src.catboost_manager import CatBoostManager
from config import RAW_TARGET_COL

def analyze_connectivity():
    print("🔍 VERİ SETİ BAĞLANTILILIK ANALİZİ BAŞLIYOR...\n")

    # 1. Veriyi Yükle
    dm = DataManager()
    dm.load_and_preprocess()
    
    # Tüm feature setini al (Hedef değişken dahil)
    df = dm.data.copy()
    
    # Sadece sayısal sütunları analiz edebiliriz (Korelasyon için)
    numeric_df = df.select_dtypes(include=[np.number])
    
    print(f"📊 Analiz edilen toplam özellik sayısı: {len(numeric_df.columns)}")

    # ---------------------------------------------------------
    # ANALİZ 1: Korelasyon Matrisi (Doğrusal Bağlantılar)
    # ---------------------------------------------------------
    print("\n[1/2] Korelasyon Matrisi Hesaplanıyor...")
    # Hedef değişkene göre en yüksek korelasyonu olan ilk 30 özelliği seçelim
    correlations = numeric_df.corr()[RAW_TARGET_COL].abs().sort_values(ascending=False)
    top_corr_features = correlations.head(31).index # Kendisi dahil 31
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(numeric_df[top_corr_features].corr(), annot=False, cmap='RdBu_r', center=0)
    plt.title(f"Top 30 Özellik - Korelasyon Isı Haritası (Hedef: {RAW_TARGET_COL})")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # ANALİZ 2: Model Bazlı Önem (Karmaşık/Mevsimsel Bağlantılar)
    # ---------------------------------------------------------
    print("\n[2/2] CatBoost ile Model Bazlı Önem Hesaplanıyor...")
    
    # Train-test split (DataManager'dan hazır alalım)
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    cat = CatBoostManager()
    cat.train_model(X_train, y_train, X_test, y_test)
    
    # Önem skorlarını al
    imp_df = cat.get_feature_importance(max_num=40)
    
    # ---------------------------------------------------------
    # ÖZET RAPOR
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📈 ANALİZ ÖZETİ")
    print("="*50)
    
    print(f"\n✅ Hedef Değişkenle En Güçlü Doğrusal İlişkisi Olanlar (Korelasyon):")
    print(correlations.head(10))
    
    print(f"\n✅ Modelin Karar Verirken En Çok Güvendiği Özellikler (CatBoost):")
    print(imp_df.head(10))
    
    # Kritik Tespit: Gereksiz özellikler
    low_importance = imp_df[imp_df['Importance'] < 0.01]
    if not low_importance.empty:
        print(f"\n⚠️ DİKKAT: {len(low_importance)} özelliğin model üzerinde etkisi %0.01'den az!")
        print("Bu özellikler ANN modelinde gürültü yaratıyor olabilir.")

if __name__ == "__main__":
    # Görselleştirme ayarları
    plt.style.use('dark_background') # İstersen 'ggplot' yapabilirsin
    analyze_connectivity()