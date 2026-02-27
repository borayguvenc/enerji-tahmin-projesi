import os
import pandas as pd
import numpy as np

def save_detailed_results(y_true, predictions_dict, project_root, filename="Detayli_Analiz_Raporu.xlsx"):
    """
    Modellerin tahmin sonuçlarını, gerçek değerleri ve sapmaları içeren
    detaylı bir Excel raporu oluşturur.

    Parametreler:
    - y_true: Gerçek hedef değerleri (Series veya Array)
    - predictions_dict: {'ModelAdi': tahmin_arrayi, ...} formatında sözlük
    - project_root: Projenin ana dizini (Raporun nereye kaydedileceğini bulmak için)
    - filename: Kaydedilecek dosya adı
    """
    
    # Rapor klasörünü oluştur
    output_dir = os.path.join(project_root, 'Reports')
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    print(f"\n[Raporlayıcı] Excel raporu hazırlanıyor: {filename}...")

    try:
        # Excel Writer Başlat (openpyxl motoru ile)
        with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
            
            # Sözlükteki her model için döngü (XGB, LGBM, Cat, Ensemble...)
            for model_name, preds in predictions_dict.items():
                
                # 1. Ana DataFrame'i oluştur (Index olarak tarihleri kullanır)
                df_result = pd.DataFrame(index=y_true.index)
                
                # 2. Sütunları Doldur
                df_result['Gerçek_Tüketim'] = y_true
                df_result['Tahmin_Edilen'] = preds
                
                # 3. Sapma Analizi
                # Sapma: Gerçek - Tahmin (Pozitifse eksik tahmin, Negatifse fazla tahmin)
                df_result['Sapma (Fark)'] = df_result['Gerçek_Tüketim'] - df_result['Tahmin_Edilen']
                
                # Mutlak Hata (Analiz kolaylığı için)
                df_result['Mutlak_Hata'] = df_result['Sapma (Fark)'].abs()
                
                # Yüzdesel Hata (Sıfıra bölünme korumalı)
                epsilon = 1e-10
                df_result['Hata_Oranı (%)'] = (df_result['Mutlak_Hata'] / (df_result['Gerçek_Tüketim'] + epsilon)) * 100

                # 4. Sayfaya Yaz (Sheet ismi modelin adı olur)
                sheet_name = model_name[:30]
                df_result.to_excel(writer, sheet_name=sheet_name)
                
        print(f"[Raporlayıcı] ✅ Rapor başarıyla kaydedildi: {full_path}")
        
    except Exception as e:
        print(f"[Raporlayıcı] ❌ Rapor kaydedilirken hata oluştu: {e}")