import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Matplotlib stil ayarları (Grafikler daha profesyonel görünsün)
plt.style.use('ggplot')
sns.set_context("talk") # Yazıları büyütür

def plot_performance(file_path, sheet_name='Grand_Ensemble', start_date=None, end_date=None):
    """
    Excel raporundan veriyi okur ve belirtilen tarih aralığını çizer.
    
    Parametreler:
    - file_path: Excel dosyasının yolu
    - sheet_name: Hangi modelin çizileceği (Grand_Ensemble, XGBoost_Detail vb.)
    - start_date: '2024-03-01' gibi başlangıç tarihi (Opsiyonel)
    - end_date: '2024-03-31' gibi bitiş tarihi (Opsiyonel)
    """
    
    print(f"📂 Dosya okunuyor: {file_path} ({sheet_name})...")
    
    try:
        # Excel'i oku (Tarih sütununu index yapıyoruz)
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0, parse_dates=True)
        
        # Sütun isimlerini kontrol et (Reporter.py'da ne verdiysek o olmalı)
        # Genelde: 'Gerçek_Tüketim', 'Tahmin_Edilen'
        col_actual = 'Gerçek_Tüketim' if 'Gerçek_Tüketim' in df.columns else df.columns[0]
        col_pred = 'Tahmin_Edilen' if 'Tahmin_Edilen' in df.columns else df.columns[1]
        
        # Tarih Filtresi (Eğer belirtildiyse)
        if start_date and end_date:
            df = df.loc[start_date:end_date]
            title_suffix = f"({start_date} - {end_date})"
        else:
            title_suffix = "(Tüm Dönem)"

        if df.empty:
            print("❌ HATA: Belirtilen tarih aralığında veri bulunamadı!")
            return

        # --- GRAFİK ÇİZİMİ ---
        # 2 satırlı bir grafik oluşturuyoruz (Üstte Tahmin, Altta Hata)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # 1. ÜST GRAFİK: Gerçek vs Tahmin
        ax1.plot(df.index, df[col_actual], label='Gerçek Tüketim', color='black', alpha=0.6, linewidth=2)
        ax1.plot(df.index, df[col_pred], label='Model Tahmini', color='#e74c3c', alpha=0.8, linewidth=1.5, linestyle='--')
        ax1.set_title(f'{sheet_name} Performansı {title_suffix}', fontsize=16)
        ax1.set_ylabel('Elektrik Tüketimi (MWh)')
        ax1.legend(loc='upper right')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        # 2. ALT GRAFİK: Sapma (Residuals)
        # Fark = Gerçek - Tahmin
        residuals = df[col_actual] - df[col_pred]
        
        # Hata çizimi
        ax2.plot(df.index, residuals, color='blue', alpha=0.6, linewidth=1)
        ax2.fill_between(df.index, residuals, 0, where=(residuals > 0), color='green', alpha=0.3, label='Eksik Tahmin (Under)')
        ax2.fill_between(df.index, residuals, 0, where=(residuals < 0), color='red', alpha=0.3, label='Fazla Tahmin (Over)')
        
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')
        ax2.set_ylabel('Hata Miktarı (MWh)')
        ax2.set_xlabel('Tarih')
        ax2.legend(loc='upper right', fontsize='small')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")

# Terminalden test etmek için:
if __name__ == "__main__":
    # Proje ana dizinini bul
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    report_path = os.path.join(project_root, 'Reports', 'YILLIK_DEV_RAPOR.xlsx')
    
    # KULLANIM ÖRNEĞİ:
    # Sadece problemli Mart ayını çizdirelim
    plot_performance(report_path, sheet_name='Grand_Ensemble', start_date='2025-07-01', end_date='2025-07-30')