import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Proje dizinini ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.data_manager import DataManager

def plot_error_vs_temp(report_path, sheet_name='Grand_Ensemble', target_month=3):
    print("1. Veri ve Rapor Yükleniyor...")
    
    # A. Orijinal Veriyi Yükle (Sıcaklık için)
    dm = DataManager()
    dm.load_and_preprocess()
    df_features = dm.data
    
    # Sıcaklık sütununu otomatik bul (Menteşe veya Ortalama)
    temp_col = next((c for c in df_features.columns if 'Hissedilen' in c and 'Mentese' in c), None)
    if not temp_col:
        temp_col = next((c for c in df_features.columns if 'Hissedilen' in c), None)
    
    print(f"   -> Kullanılan Sıcaklık Sütunu: {temp_col}")

    # B. Excel Raporunu Yükle (Hata ve Tahmin için)
    df_preds = pd.read_excel(report_path, sheet_name=sheet_name, index_col=0, parse_dates=True)
    
    # C. İki Veriyi Birleştir (Tarih İndeksine Göre)
    # Sadece ortak tarihleri alır
    df_merged = df_preds.join(df_features[[temp_col]], how='inner')
    
    # D. Sadece Hedef Ayı Seç (Örn: Mart = 3)
    df_march = df_merged[df_merged.index.month == target_month].copy()
    
    if df_march.empty:
        print("HATA: Seçilen ayda veri bulunamadı!")
        return

    # Hata Hesabı (Eğer Excel'de yoksa)
    # Pozitif Hata = Eksik Tahmin (Underestimation)
    # Negatif Hata = Fazla Tahmin (Overestimation)
    if 'Sapma (Fark)' not in df_march.columns:
        df_march['Sapma (Fark)'] = df_march['Gerçek_Tüketim'] - df_march['Tahmin_Edilen']

    # --- GRAFİK ÇİZİMİ ---
    plt.figure(figsize=(12, 8))
    
    # Scatter Plot ile Regresyon Doğrusu
    # hue='Saat' ekleyerek günün hangi saatinde olduğunu renklerle görebiliriz
    df_march['Saat'] = df_march.index.hour
    
    sns.scatterplot(
        data=df_march, 
        x=temp_col, 
        y='Sapma (Fark)', 
        hue='Saat', 
        palette='viridis', 
        alpha=0.7,
        s=60
    )
    
    # Sıfır Hattı (Mükemmel Tahmin)
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Sıfır Hata')
    
    # Regresyon Doğrusu (Trendi görmek için)
    sns.regplot(
        data=df_march, 
        x=temp_col, 
        y='Sapma (Fark)', 
        scatter=False, 
        color='red', 
        line_kws={'label': 'Trend (Eğim)'}
    )

    plt.title(f'Mart Ayı: Hata Miktarı vs. Sıcaklık Analizi\n({temp_col})', fontsize=16)
    plt.xlabel('Hissedilen Sıcaklık (°C)', fontsize=14)
    plt.ylabel('Hata Miktarı (MWh)\n(Pozitif=Eksik Tahmin, Negatif=Fazla Tahmin)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bölgeleri Açıkla
    plt.text(df_march[temp_col].min(), df_march['Sapma (Fark)'].max(), "SOL ÜST KÖŞE:\nSoğukta Eksik Tahmin\n(Isıtma Sorunu)", color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Rapor dosyasının tam yolu
    report_file = os.path.join(project_root, 'Reports', 'YILLIK_DEV_RAPOR.xlsx')
    
    # Mart ayı (3) için çalıştır
    plot_error_vs_temp(report_file, sheet_name='Grand_Ensemble', target_month=3)