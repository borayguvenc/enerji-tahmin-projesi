import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Proje kök dizinine erişim
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_manager import DataManager
from config import RAW_TARGET_COL

def analyze_and_kill_features(threshold=0.95):
    print("📊 Veri Yükleniyor ve Analiz Başlıyor...")
    
    dm = DataManager()
    dm.load_and_preprocess()
    df = dm.data
    
    # Sadece sayısal sütunları al (Kategoriklerle korelasyon bakılmaz)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Hedef sütunu koru, onu silmeyelim
    if RAW_TARGET_COL not in numeric_df.columns:
        print(f"HATA: Hedef sütun '{RAW_TARGET_COL}' bulunamadı!")
        return

    # 1. HEDEF İLE KORELASYON (Target Correlation)
    # Hangi özellikler hedefi tahmin etmede en etkili?
    target_corr = numeric_df.corrwith(numeric_df[RAW_TARGET_COL]).abs().sort_values(ascending=False)
    
    print("\n🏆 HEDEFLE EN YÜKSEK İLİŞKİLİ İLK 10 ÖZELLİK:")
    print(target_corr.head(10))
    
    print("\n💀 HEDEFLE EN DÜŞÜK İLİŞKİLİ 10 ÖZELLİK (Çöp Adayları):")
    print(target_corr.tail(10))

    # 2. ÇOKLU BAĞLANTI ANALİZİ (Multicollinearity)
    # Birbiriyle çok yüksek korelasyonlu özellikleri bul
    print(f"\n⚔️ KORELASYON SAVAŞI BAŞLIYOR (Eşik: {threshold})")
    print("Mantık: İki özellik birbirine %{:.0f}'ten fazla benziyorsa, hedefle ilişkisi daha zayıf olan SİLİNECEK.\n".format(threshold*100))
    
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    
    for column in upper.columns:
        # Bu sütunla yüksek korelasyonlu olan diğer sütunları bul
        high_corr_cols = upper.index[upper[column] > threshold].tolist()
        
        for feature_b in high_corr_cols:
            feature_a = column
            
            # Hangisi hedefle daha ilişkili?
            corr_a = target_corr[feature_a]
            corr_b = target_corr[feature_b]
            
            print(f"⚠️ ÇATIŞMA: '{feature_a}' vs '{feature_b}' (Benzerlik: {upper.loc[feature_b, feature_a]:.4f})")
            
            if corr_a > corr_b:
                print(f"   -> KAZANAN: {feature_a} (Corr: {corr_a:.4f})")
                print(f"   -> SİLİNEN: {feature_b} (Corr: {corr_b:.4f})")
                to_drop.append(feature_b)
            else:
                print(f"   -> KAZANAN: {feature_b} (Corr: {corr_b:.4f})")
                print(f"   -> SİLİNEN: {feature_a} (Corr: {corr_a:.4f})")
                to_drop.append(feature_a)
                
    # Tekrarlananları temizle
    to_drop = list(set(to_drop))
    
    # Hedef sütunu yanlışlıkla listeye girdiyse kurtar
    if RAW_TARGET_COL in to_drop:
        to_drop.remove(RAW_TARGET_COL)

    print(f"\n\n🗑️ TOPLAM {len(to_drop)} ÖZELLİK İÇİN İDAM KARARI VERİLDİ:")
    print("-------------------------------------------------------------")
    for feature in to_drop:
        print(f"{feature}")
        
    print("\n✅ TAVSİYE: Bu listeyi kopyalayıp data_manager.py içinde 'drop' edebilirsin.")

if __name__ == "__main__":
    analyze_and_kill_features(threshold=0.95) # %95 ve üzeri benzerlikleri affetme