import pandas as pd
import numpy as np
import os 
import sys

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
        # 4. YENİ ÖZELLİK MÜHENDİSLİĞİ (Gelişmiş Hava Durumu)
        # ----------------------------------------------------
        BASE_TEMP_COL = 'Hissedilen_Sıcaklık-MUGLA_MenteseCenter_OpenMeteo'
        
        if BASE_TEMP_COL in df.columns:
            
            # 1A. Sıcaklığın Karesi (Non-Linear U-Eğrisi)
            # Konfor sıcaklığını 18°C kabul edip uzaklığı hesaplıyoruz.
            # Tüketimin, 18'den uzaklaştıkça artacağını vurgular.
            df['Temp_Squared_18'] = (df[BASE_TEMP_COL] - 18) ** 2
            
            # 1B. Termal Atalet (Sıcaklık Lag'leri)
            # Binaların geç tepki verme süresini yakalar.
            for lag in [3, 6, 12]:
                new_col_name = f'Temp_Lag{lag}h'
                
                # 'shift' fonksiyonu, veriyi belirtilen adım kadar aşağı kaydırarak
                # geçmişteki değeri şimdiki satıra getirir.
                df[new_col_name] = df[BASE_TEMP_COL].shift(lag)
                
            print("   -> Temp_Squared_18, Temp_Lag3h/6h/12h eklendi.")
        else:
            print(f"   -> UYARI: {BASE_TEMP_COL} sütunu bulunamadı, Termal Özellikler oluşturulamadı.")

        # ----------------------------------------------------
        # 5. Kategorik Veri İşleme
        # Özel Günler -> Category
        if 'ÖzelGün_Adı' in df.columns:
            print("[DataManager] Converting 'ÖzelGün_Adı' to category...")
            df['ÖzelGün_Adı'] = df['ÖzelGün_Adı'].astype('category')

        # Is_lockdown -> Genelde 0/1 olur, int veya bool kalabilir.
        if 'Is_lockdown' in df.columns:
             df['Is_lockdown'] = df['Is_lockdown'].astype(int)

        # dff: Difference (Fark)
        """
        df['Load_Diff_1h'] = df[RAW_TARGET_COL].diff(periods=1).shift(24) 
        df['Load_Diff_24h'] = df[RAW_TARGET_COL].diff(periods=24).shift(24)
        df['Temp_Slope_3h'] = df[BASE_TEMP_COL].diff(periods=3)
        """

        # ----------------------------------------------------
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