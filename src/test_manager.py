import pandas as pd
from data_manager import DataManager  # Dosya adının data_manager.py olduğunu varsayıyorum

def run_test():
    print("=== TEST BAŞLIYOR ===")
    
    # 1. Sınıfı Başlat
    dm = DataManager()
    
    # 2. Yükleme ve Ön İşlemeyi Çalıştır
    print("\n[Adım 1] load_and_preprocess() çalıştırılıyor...")
    df = dm.load_and_preprocess()
    
    # --- KONTROL NOKTALARI ---
    
    # A. Sütun Kontrolü (Sin/Cos gitti mi?)
    print("\n--- A. Sütun Kontrolü ---")
    bad_cols = [c for c in df.columns if 'Sin' in c or 'Cos' in c]
    if not bad_cols:
        print("✅ BAŞARILI: Sin/Cos sütunları temizlenmiş.")
    else:
        print(f"❌ HATA: Sin/Cos sütunları hala duruyor: {bad_cols}")
        
    # B. Veri Tipi Kontrolü (XGBoost uyumlu mu?)
    print("\n--- B. Veri Tipi Kontrolü ---")
    print(f"Saat Tipi: {df['Saat'].dtype} (Beklenen: int32 veya int64)")
    print(f"ÖzelGün Tipi: {df['ÖzelGün_Adı'].dtype} (Beklenen: category)")
    
    if pd.api.types.is_integer_dtype(df['Saat']):
        print("✅ BAŞARILI: Saat sütunu Integer formatında.")
    else:
        print("❌ HATA: Saat sütunu Integer değil!")

    if isinstance(df['ÖzelGün_Adı'].dtype, pd.CategoricalDtype):
        print("✅ BAŞARILI: ÖzelGün_Adı Category formatında.")
    else:
        print("❌ HATA: ÖzelGün_Adı Category yapılmamış!")

    # C. İndeks Kontrolü
    print("\n--- C. İndeks (Zaman) Kontrolü ---")
    print(df.index[:3])
    if isinstance(df.index, pd.DatetimeIndex):
        print("✅ BAŞARILI: İndeks DatetimeIndex formatında.")
    else:
        print("❌ HATA: İndeks tarih formatında değil.")

    # 3. Train/Test Ayrımı Testi
    print("\n[Adım 2] get_train_test_split() çalıştırılıyor...")
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    print("\n--- D. Boyut Kontrolü ---")
    print(f"Train Seti: {X_train.shape}")
    print(f"Test Seti : {X_test.shape}")
    
    # Veri sızıntısı (Leakage) kontrolü
    if X_train.index.max() < X_test.index.min():
        print("✅ BAŞARILI: Tarihsel sıralama doğru (Train, Test'ten önce geliyor).")
    else:
        print("❌ HATA: Train ve Test tarihleri karışmış!")

    # 4. Son Özellik Listesi
    print("\n--- E. Modele Girecek Sütunlar ---")
    print(X_train.columns.tolist())

if __name__ == "__main__":
    run_test()