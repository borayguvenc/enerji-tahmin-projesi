import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)            
sys.path.append(project_root)

from src.data_manager import DataManager


def analyse_weather_quality():
    # 1. Veriyi yükle
    dm = DataManager()
    df = dm.load_and_preprocess() # DataManager içindeki tüm temizlikler dahil gelir
    
    # 2. Karşılaştırılacak sütun çiftlerini bul
    fc_cols = [c for c in df.columns if c.endswith('_fc')]
    
    results = []
    
    print("\n🌡️ HAVA DURUMU TAHMİN KALİTESİ ANALİZİ")
    print("-" * 50)
    
    for f_col in fc_cols:
        a_col = f_col.replace('_fc', '_actual')
        if a_col in df.columns:
            # Hataları hesapla
            error = df[a_col] - df[f_col]
            mae = error.abs().mean()
            bias = error.mean() # Pozitifse tahmin düşük, negatifse tahmin yüksek kalmış
            correlation = df[a_col].corr(df[f_col])
            
            results.append({
                'İstasyon': f_col.replace('_fc', ''),
                'MAE (Derece Sapma)': round(mae, 2),
                'Bias (Sistematik)': round(bias, 2),
                'Korelasyon': round(correlation, 4)
            })

    # Sonuçları tablo olarak göster
    results_df = pd.DataFrame(results).sort_values(by='MAE (Derece Sapma)')
    print(results_df.to_string(index=False))

    # 3. GÖRSEL ANALİZ (Örnek bir istasyon için)
    sample_station = results[0]['İstasyon']
    plt.figure(figsize=(15, 6))
    
    # Son 1 haftalık (168 saat) kesit alalım
    recent_data = df.tail(168)
    plt.plot(recent_data.index, recent_data[sample_station + '_actual'], label='Gerçekleşen', marker='o', alpha=0.7)
    plt.plot(recent_data.index, recent_data[sample_station + '_fc'], label='Tahmin (Open-Meteo)', linestyle='--', marker='x', alpha=0.7)
    
    plt.title(f"Son 1 Hafta Kıyaslaması: {sample_station}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. SAATLİK HATA ANALİZİ (Isı Haritası)
    df['Error'] = df[results[0]['İstasyon'] + '_actual'] - df[results[0]['İstasyon'] + '_fc']
    hourly_error = df.groupby('Saat')['Error'].mean()
    
    plt.figure(figsize=(10, 5))
    hourly_error.plot(kind='bar', color='orange')
    plt.axhline(0, color='black', linestyle='-')
    plt.title("Günün Saatlerine Göre Ortalama Sıcaklık Sapması (Bias)")
    plt.ylabel("Derece Farkı (Actual - Forecast)")
    plt.show()

if __name__ == "__main__":
    analyse_weather_quality()