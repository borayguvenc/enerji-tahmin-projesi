import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 1. YOL AYARLARI (SRC KLASÖRÜNE ERİŞİM)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

# ---------------------------------------------------------
# 2. İMPORTLAR
# ---------------------------------------------------------
from config import (NUM_OF_SPLITS, TEST_SIZE, RAW_TARGET_COL)
from src.data_manager import DataManager
from src.evaluator import Evaluator

# Model Yöneticileri
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost

# ---------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, model_name="Ensemble"):
    """Manuel metrik hesaplama fonksiyonu (Ensemble modu için)"""
    epsilon = 1e-10
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    print(f"\n=== {model_name} PERFORMANCE ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : %{mape:.2f}")
    print("==============================\n")
    return mape


def main():
    print("🚀 SİSTEM BAŞLATILIYOR...\n")

    # A. MOD SEÇİMİ (Argüman veya Kullanıcı Girdisi)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='Mod: XGB, LGBM, CAT, ALL')
    args = parser.parse_args()

    if args.mode:
        mode = args.mode.upper()
    else:
        print("Hangi model çalıştırılsın?")
        print("1. XGB  (Sadece XGBoost)")
        print("2. LGBM (Sadece LightGBM)")
        print("3. CAT  (Sadece CatBoost)")
        print("4. ALL  (Grand Ensemble: XGB + LGBM + CAT)")
        
        selection = input("Seçiminiz (XGB/LGBM/CAT/ALL): ").upper()
        
        if selection in ['1', 'XGB']: mode = 'XGB'
        elif selection in ['2', 'LGBM']: mode = 'LGBM'
        elif selection in ['3', 'CAT']: mode = 'CAT'
        elif selection in ['4', 'ALL', 'BOTH']: mode = 'ALL'
        else: 
            print("Geçersiz seçim! Varsayılan olarak XGB çalıştırılıyor.")
            mode = 'XGB'

    print(f"\n[SİSTEM] Çalışma Modu: {mode}\n")

    # B. VERİ YÜKLEME VE HAZIRLIK
    dm = DataManager()
    dm.load_and_preprocess()
    
    # Train/Test Ayrımı (Son 30 gün Test, kalanı Train)
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    # Cross Validation için tüm veriyi hazırla (Evaluator kullanacak)
    feature_cols = dm.data.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
    if RAW_TARGET_COL in feature_cols: 
        feature_cols.remove(RAW_TARGET_COL)
    
    X_full = dm.data[feature_cols]
    y_full = dm.data[RAW_TARGET_COL]

    # Evaluator Başlat (Ortak kullanım)
    evaluator = Evaluator(n_splits=NUM_OF_SPLITS, test_size=TEST_SIZE)

    # --- MOD 1: XGBOOST ---
    if mode == 'XGB':
        print("\n--- XGBoost Final Training ---")
        mm = ModelManager()
        mm.train_model(X_train, y_train, X_test, y_test)
        mm.evaluate(X_test, y_test)
        # KAYIT KOMUTU:
        mm.save_model("final_xgboost.json") 

    # --- MOD 2: LIGHTGBM ---
    elif mode == 'LGBM':
        print("\n--- LightGBM Final Training ---")
        lgbm = LightGBMManager()
        lgbm.train_model(X_train, y_train, X_test, y_test)
        lgbm.evaluate(X_test, y_test)
        # KAYIT KOMUTU:
        lgbm.save_model("final_lightgbm.txt")

    # --- MOD 3: CATBOOST ---
    elif mode == 'CAT':
        print("\n--- CatBoost Final Training ---")
        cat = CatBoostManager()
        cat.train_model(X_train, y_train, X_test, y_test)
        cat.evaluate(X_test, y_test)
        # KAYIT KOMUTU:
        cat.save_model("final_catboost.cbm")

    # --- MOD 4: GRAND ENSEMBLE (HEPSİ) ---
    elif mode == 'ALL':
        print("\n--- 🚀 GRAND ENSEMBLE MODU (XGB + LGBM + CAT) ---")
        
        # 1. XGBoost
        print(">>> 1/3 XGBoost Eğitiliyor ve Kaydediliyor...")
        xgb_man = ModelManager()
        xgb_man.train_model(X_train, y_train, X_test, y_test)
        pred_xgb = xgb_man.model.predict(X_test)
        xgb_man.save_model("ensemble_xgb.json") # <-- KAYIT 1

        # 2. LightGBM
        print("\n>>> 2/3 LightGBM Eğitiliyor ve Kaydediliyor...")
        lgbm_man = LightGBMManager()
        lgbm_man.train_model(X_train, y_train, X_test, y_test)
        pred_lgbm = lgbm_man.model.predict(X_test)
        lgbm_man.save_model("ensemble_lgbm.txt") # <-- KAYIT 2

        # 3. CatBoost
        print("\n>>> 3/3 CatBoost Eğitiliyor ve Kaydediliyor...")
        cat_man = CatBoostManager()
        cat_man.train_model(X_train, y_train, X_test, y_test)
        pred_cat = cat_man.model.predict(X_test)
        cat_man.save_model("ensemble_cat.cbm") # <-- KAYIT 3

        # ORTALAMA
        final_preds = (pred_xgb + pred_lgbm + pred_cat) / 3

        # SONUÇ
        print("\n🏆 GRAND ENSEMBLE SONUÇLARI (SON AY) 🏆")
        calculate_metrics(y_test, final_preds, model_name="Triple Ensemble (XGB+LGBM+CAT)")
        
        # Meraklısına: Kim ne tahmin etti? (İlk satır örneği)
        print(f"XGBoost  İlk Tahmin : {pred_xgb[0]:.2f}")
        print(f"LightGBM İlk Tahmin : {pred_lgbm[0]:.2f}")
        print(f"CatBoost İlk Tahmin : {pred_cat[0]:.2f}")
        print(f"ORTALAMA (SONUÇ)    : {final_preds[0]:.2f}")
        print(f"GERÇEK DEĞER        : {y_test.iloc[0]:.2f}")

if __name__ == "__main__":
    main()