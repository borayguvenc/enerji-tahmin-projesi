import pandas as pd
import numpy as np
import os 
import sys
from src.smart_features import SmartFeatureEngineer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)            
sys.path.append(project_root)

from config import (
    INPUT_FILE_PATH, 
    RAW_TARGET_COL, 
    RAW_DATE_COL, 
    RAW_HOUR_COL, 
    COLS_TO_DROP,
    WARMUP_PERIOD, 
    TEST_SIZE
)

class DataManager:
    def __init__(self):
        self.data = None
        
    def load_and_preprocess(self):
        """
        Veriyi yükler, tarih ayarını yapar, kategorik dönüşümü yapar.
        """
        print(f"[DataManager] Loading data from: {INPUT_FILE_PATH}")
        df = pd.read_excel(INPUT_FILE_PATH, engine='openpyxl')


        # 2. Sayısal Dönüşümler (Virgül -> Nokta)
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Tarih ve Özel Gün dışındakileri sayıya çevir
            if col in [RAW_DATE_COL, 'ÖzelGün_Adı']:
                continue
            try:
                # Varsa virgülleri noktaya çevirip float yap
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except (ValueError, AttributeError):
                pass



        # ----------------------------------------------------
        # 3. Tarih ve Saat İşlemleri
        df[RAW_DATE_COL] = pd.to_datetime(df[RAW_DATE_COL])
        
        # Tam datetime index oluşturma
        df['Datetime'] = df[RAW_DATE_COL].dt.normalize() + pd.to_timedelta(df[RAW_HOUR_COL], unit='h')
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True) # Tarih sırasını garantiye al

        # Zaman bileşenlerinin INT olduğundan emin ol (XGBoost için kritik)
        time_features = ['Yıl', 'Ay', 'Gün', 'Saat', 'Haftanın_Günü']
        for tf in time_features:
            if tf in df.columns:
                df[tf] = df[tf].astype(int)
        
        
        # ----------------------------------------------------
        # [ADIM 2] OKUL TATİLLERİNİ AYRIŞTIRMA (Yaz vs Kış)
        # ----------------------------------------------------
        print("[DataManager] Sömestr ve Yaz tatilleri işleniyor...")

        # 1. SÖMESTR TATİLLERİ (Kış Karakteristiği)
        semester_ranges = [
            ('2018-01-22', '2018-02-04'), ('2019-01-21', '2019-02-03'),
            ('2020-01-20', '2020-02-02'), ('2021-01-25', '2021-02-14'),
            ('2022-01-24', '2022-02-06'), ('2023-01-23', '2023-02-19'), # Deprem dahil
            ('2024-01-22', '2024-02-04'), ('2025-01-20', '2025-02-02')
        ]
        
        df['Is_Semester'] = 0
        for start, end in semester_ranges:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'Is_Semester'] = 1

        # 2. YAZ TATİLLERİ (Turizm/Klima Karakteristiği)
        summer_ranges = [
            ('2018-06-09', '2018-09-16'), ('2019-06-15', '2019-09-08'),
            ('2020-03-16', '2020-09-20'), 
            ('2021-07-03', '2021-09-05'), ('2022-06-18', '2022-09-11'),
            ('2023-06-17', '2023-09-10'), ('2024-06-15', '2024-09-08'),
            ('2025-06-21', '2025-09-08')
        ]

        df['Is_Summer_Break'] = 0
        for start, end in summer_ranges:
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, 'Is_Summer_Break'] = 1
            
        df['Is_Semester'] = df['Is_Semester'].astype(int)
        df['Is_Summer_Break'] = df['Is_Summer_Break'].astype(int)



        
        # ----------------------------------------------------
        # 4. YENİ ÖZELLİK MÜHENDİSLİĞİ (Gelişmiş Hava Durumu & Aggregation)
        # ----------------------------------------------------
        print("[DataManager] İl bazlı sıcaklık ortalamaları ve termal özellikler hesaplanıyor...")

        # A. İL BAZLI ORTALAMALAR (AGGREGATION)
        # Tek tek istasyonlar yerine il genelini temsil eden ortalamaları alıyoruz.
        province_map = {
            'MUGLA': 'Hissedilen_Sıcaklık_Mean_MUGLA',
            'DNZ':   'Hissedilen_Sıcaklık_Mean_DNZ',
            'AYD':   'Hissedilen_Sıcaklık_Mean_AYD'
        }

        cols_to_remove = []

        # Her il için döngü
        for province_code, new_col_name in province_map.items():
            # O ilin kodunu ve 'Hissedilen_Sıcaklık' ismini içeren tüm sütunları bul
            relevant_cols = [c for c in df.columns if province_code in c and 'Hissedilen_Sıcaklık' in c]
            
            if relevant_cols:
                # Satır bazında (axis=1) ortalama alarak tek sütuna indir
                df[new_col_name] = df[relevant_cols].mean(axis=1)
                
                # Orijinal kalabalık sütunları silinecekler listesine ekle
                cols_to_remove.extend(relevant_cols)
                print(f"   -> {new_col_name} oluşturuldu ({len(relevant_cols)} istasyon birleştirildi).")

        # B. TEMİZLİK (Gürültü Azaltma)
        # Orijinal 14 sütunu kaldırıp yerine 3 temiz sütun bırakıyoruz.
        if cols_to_remove:
            df.drop(columns=cols_to_remove, inplace=True)
        
        # C. TERMAL ÖZELLİKLER İÇİN "BAZ" SÜTUN SEÇİMİ
        # Eskiden 'MenteseCenter' kullanıyorduk, artık 'Muğla Ortalaması'nı kullanacağız.
        # Eğer Muğla yoksa Denizli'yi, o da yoksa Aydın'ı dener.
        available_means = [
            'Hissedilen_Sıcaklık_Mean_MUGLA', 
            'Hissedilen_Sıcaklık_Mean_DNZ', 
            'Hissedilen_Sıcaklık_Mean_AYD'
        ]
        
        # Listeden veri setinde var olan ilk sütunu seç
        BASE_TEMP_COL = next((col for col in available_means if col in df.columns), None)
        
        if BASE_TEMP_COL:
            print(f"   -> Termal özellikler için baz alınan sütun: {BASE_TEMP_COL}")
            
            # 1. Sıcaklığın Karesi (U-Eğrisi)
            # Konfor sıcaklığından (18) uzaklaştıkça değer büyür (Klima/Isıtma etkisi)
            df['Temp_Squared_18'] = (df[BASE_TEMP_COL] - 18) ** 2
            
            # 2. Termal Atalet (Sıcaklık Lag'leri)
            # Binaların ısınma/soğuma gecikmesi
            for lag in [3, 6, 12]:
                df[f'Temp_Lag{lag}h'] = df[BASE_TEMP_COL].shift(lag)
            
            # 3. [ÖNEMLİ] LAG TUZAĞINI KIRAN "DELTA" ÖZELLİKLERİ
            # Modelin "Dün hava nasıldı?" yerine "Hava dünden ne kadar değişti?" sorusunu cevaplaması için.
            df['Temp_Diff_24h'] = df[BASE_TEMP_COL] - df[BASE_TEMP_COL].shift(24)
            df['Temp_Diff_3h'] = df[BASE_TEMP_COL].diff(3)

            print("   -> Temp_Squared, Lags ve Temp_Diff (Delta) özellikleri eklendi.")
        else:
            print("   -> UYARI: Hiçbir sıcaklık ortalaması oluşturulamadı! Termal özellikler atlanıyor.")

            
        # ----------------------------------------------------
        # 5. Kategorik Veri İşleme
        # Özel Günler -> Category
        if 'ÖzelGün_Adı' in df.columns:
            print("[DataManager] Converting 'ÖzelGün_Adı' to category...")
            df['ÖzelGün_Adı'] = df['ÖzelGün_Adı'].astype('category')


        binary_flags = ['Is_Ramadan', 'Is_Sahur', 'Is_lockdown', 'Ramazan_Bayram','Yilbasi',"Kurban_Bayram","Secim_Gunu", "Milli_Bayram"] 

        print(f"[DataManager] Binary sütunlar (0/1) işleniyor: {binary_flags}")

        for col in binary_flags:
            if col in df.columns:
                # 1. Her ihtimale karşı eksik varsa 0 yap (Sen eksik yok dedin ama güvenliktir)
                df[col] = df[col].fillna(0)
                
                # 2. Tipini 'int' (Tamsayı) yap.
                df[col] = df[col].astype(int)
            else:
                print(f"   -> UYARI: '{col}' sütunu Excel'de bulunamadı! İsmi doğru yazdın mı?")

        # Son 3 günün aynı saatinin ortalaması (Dün 14:00 + Evvelsi 14:00 + ...)
        # (Lag24 + Lag48 + Lag72) / 3
        df['Mean_Last_3_Days_Same_Hour'] = (
            df[RAW_TARGET_COL].shift(24) + 
            df[RAW_TARGET_COL].shift(48) + 
            df[RAW_TARGET_COL].shift(72)
        ) / 3


        # ----------------------------------------------------
        # 6. AKILLI KOMŞU ÖZELLİĞİ (SMART NEIGHBOR FEATURE)
        # ----------------------------------------------------
        """
        smart_engineer = SmartFeatureEngineer(df)
        
        # Aday sıcaklık sütunlarını verelim (Varsa ortalamayı, yoksa Menteşe'yi kullanır)
        temp_candidates = [
            'Hissedilen_Sıcaklık_Mean_MUGLA', 
            'Hissedilen_Sıcaklık-MUGLA_MenteseCenter_OpenMeteo'
        ]
        
        # İşlemi yap ve df'i güncelle
        df = smart_engineer.add_smart_neighbor_feature(
            target_col=RAW_TARGET_COL, 
            temp_col_candidates=temp_candidates
        )
        """



        # 6. Config'den Gelen Gereksiz Sütunları Atma
        if COLS_TO_DROP:
            print(f"[DataManager] Dropping columns from config: {COLS_TO_DROP}")
            existing_drop_cols = [c for c in COLS_TO_DROP if c in df.columns]
            df.drop(columns=existing_drop_cols, inplace=True)

        

        # 7. Eksik Verileri Çıkarma
        print(f"[DataManager] Dropping warm-up period ({WARMUP_PERIOD} rows)...")
        df = df.iloc[WARMUP_PERIOD:] 
        df.dropna(inplace=True)

        self.data = df
        print(f"[DataManager] Preprocessing complete. Shape: {df.shape}")
        
        # Veri tiplerini kontrol için yazdır
        print(df.dtypes)
        
        return self.data

    def get_train_test_split(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        # Test seti son n satır
        split_idx = len(self.data) - TEST_SIZE
        train_df = self.data.iloc[:split_idx]
        test_df = self.data.iloc[split_idx:]

        print(f"[DataManager] Split Done. Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Feature Selection
        # number: int, float | category: category | bool: bool
        feature_cols = self.data.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
        
        if RAW_TARGET_COL in feature_cols:
            feature_cols.remove(RAW_TARGET_COL)
            
        print(f"[Features Used]: {feature_cols}")

        X_train = train_df[feature_cols]
        y_train = train_df[RAW_TARGET_COL]
        X_test = test_df[feature_cols]
        y_test = test_df[RAW_TARGET_COL]

        return X_train, y_train, X_test, y_test