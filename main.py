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
from src.reporter import save_detailed_results

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
    print(" SİSTEM BAŞLATILIYOR...\n")

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
        
        # 1. CROSS VALIDATION (12 Aylık Tahminleri Topla)
        print("\n[Adım 1] Tüm Yıl İçin Cross Validation Çalıştırılıyor...")
        
        # ARTIK 2 ŞEY DÖNÜYOR: Skorlar VE Tüm Tahminler
        scores, full_year_results = evaluator.run_cross_validation(X_full, y_full, model_type='ALL')
        
        evaluator.print_summary(scores)

        # ---------------------------------------------------------
        # 2. TÜM YIL RAPORU (Excel) 📊
        # ---------------------------------------------------------
        print("\n[Adım 2] Yıllık Detaylı Rapor Hazırlanıyor...")
        
        # Reporter'a göndermek için paketle
        # full_year_results DataFrame'inden sütunları çekiyoruz
        predictions_pack = {
            'Grand_Ensemble': full_year_results['Ensemble_Pred'],
            'XGBoost_Detail': full_year_results['XGB_Pred'],
            'LightGBM_Detail': full_year_results['LGBM_Pred'],
            'CatBoost_Detail': full_year_results['CAT_Pred']
        }
        
        # Gerçek değerler (Index tarih olduğu için eşleşir)
        y_true_full = full_year_results['Actual']
        
        # Kaydet
        save_detailed_results(
            y_true=y_true_full,
            predictions_dict=predictions_pack,
            project_root=current_dir,
            filename="YILLIK_DEV_RAPOR.xlsx"
        )
    

if __name__ == "__main__":
    main()