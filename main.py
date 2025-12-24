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
from src.catboost_bayram_manager import CatBoostBayramManager
from src.experiment_logger import ExperimentLogger  # <--- LOGGER
from src.reporter import save_detailed_results      # <--- EXCEL RAPOR

# Model Yöneticileri
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost

def main():
    print("🚀 SİSTEM BAŞLATILIYOR...\n")

    # A. MOD SEÇİMİ
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
    
    # Train/Test Ayrımı (Final Training için)
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    # Cross Validation Verisi (Evaluator için)
    feature_cols = dm.data.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
    if RAW_TARGET_COL in feature_cols: 
        feature_cols.remove(RAW_TARGET_COL)
    
    X_full = dm.data[feature_cols]
    y_full = dm.data[RAW_TARGET_COL]

    # ---------------------------------------------------------
    # C. ORTAK ARAÇLARIN HAZIRLANMASI (Logger & Evaluator)
    # ---------------------------------------------------------
    # Evaluator'ı başlat
    evaluator = Evaluator(n_splits=NUM_OF_SPLITS, test_size=TEST_SIZE)
    
    # Logger'ı başlat
    logger = ExperimentLogger(project_root=current_dir)
    
    # Config Paketini Hazırla (Bunu tüm modeller ortak kullanacak)
    config_pack = {
        "NUM_OF_SPLITS": NUM_OF_SPLITS,
        "TEST_SIZE": TEST_SIZE,
        "RAW_TARGET_COL": RAW_TARGET_COL,
        "FEATURE_COUNT": len(feature_cols)
    }

    # ---------------------------------------------------------
    # D. MODEL AKIŞLARI
    # ---------------------------------------------------------

    # --- MOD 1: XGBOOST ---
    if mode == 'XGB':
        print("\n--- 🔍 XGBoost Cross Validation ---")
        scores, _ = evaluator.run_cross_validation(X_full, y_full, model_type='XGB')
        evaluator.print_summary(scores)
        
        # LOGLAMA (Solo)
        logger.log_experiment(
            model_name="XGBoost_Solo",
            mape_scores=scores,
            config_dict=config_pack,
            notes="Single Model Run"
        )

        print("\n--- 💾 XGBoost Final Training & Saving ---")
        mm = ModelManager()
        mm.train_model(X_train, y_train, X_test, y_test)
        mm.evaluate(X_test, y_test)
        mm.save_model("final_xgboost.json") 

    # --- MOD 2: LIGHTGBM ---
    elif mode == 'LGBM':
        print("\n--- 🔍 LightGBM Cross Validation ---")
        scores, _ = evaluator.run_cross_validation(X_full, y_full, model_type='LGBM')
        evaluator.print_summary(scores)
        
        # LOGLAMA (Solo)
        logger.log_experiment(
            model_name="LightGBM_Solo",
            mape_scores=scores,
            config_dict=config_pack,
            notes="Single Model Run"
        )

        print("\n--- 💾 LightGBM Final Training & Saving ---")
        lgbm = LightGBMManager()
        lgbm.train_model(X_train, y_train, X_test, y_test)
        lgbm.evaluate(X_test, y_test)
        lgbm.save_model("final_lightgbm.txt")

    # --- MOD 3: CATBOOST ---
    elif mode == 'CAT':
        print("\n--- 🔍 CatBoost Cross Validation ---")
        scores, _ = evaluator.run_cross_validation(X_full, y_full, model_type='CAT')
        evaluator.print_summary(scores)
        
        # LOGLAMA (Solo)
        logger.log_experiment(
            model_name="CatBoost_Solo",
            mape_scores=scores,
            config_dict=config_pack,
            notes="Single Model Run"
        )

        print("\n--- 💾 CatBoost Final Training & Saving ---")
        cat = CatBoostManager()
        cat.train_model(X_train, y_train, X_test, y_test)
        cat.evaluate(X_test, y_test)
        cat.save_model("final_catboost.cbm")

    # --- MOD 4: GRAND ENSEMBLE (HEPSİ) ---
    elif mode == 'ALL':
        print("\n--- 🚀 GRAND ENSEMBLE MODU (XGB + LGBM + CAT) ---")
        
        # 1. CV Analizi
        scores, full_year_results = evaluator.run_cross_validation(X_full, y_full, model_type='ALL')
        evaluator.print_summary(scores)

        # 2. LOGLAMA
        print("\n[Logger] Sonuçlar veritabanına işleniyor...")
        logger.log_experiment(
            model_name="Grand_Ensemble",
            mape_scores=scores,
            config_dict=config_pack,
            notes="Ensemble Run"
        )

        
        # 3. Yıllık Rapor (Excel)
        print("\n[Rapor] Yıllık Detaylı Excel Hazırlanıyor...")
        
        predictions_pack = {
            'Grand_Ensemble': full_year_results['Ensemble_Pred'],
            'XGBoost_Detail': full_year_results['XGB_Pred'],
            'LightGBM_Detail': full_year_results['LGBM_Pred'],
            'CatBoost_Detail': full_year_results['CAT_Pred']
        }
        
        y_true_full = full_year_results['Actual']

        save_detailed_results(
            y_true=y_true_full,
            predictions_dict=predictions_pack,
            project_root=current_dir,
            filename="YILLIK_DENEME_BAYRAM_AFTER.xlsx"
        )
        
    elif mode == 'SNIPER':
        # --- MOD 5: SNIPER (BAYRAM) MODELİ ---
        print("\n--- 🦅 Sniper Model Devrede ---")

        # 1. Evaluator'ı 'SNIPER' moduyla çalıştır
        scores, full_year_results = evaluator.run_cross_validation(X_full, y_full, model_type='SNIPER')
        evaluator.print_summary(scores)
        
        # 2. Loglama
        logger.log_experiment(
            model_name="Sniper_Bayram_Model", 
            mape_scores=scores, 
            config_dict=config_pack, 
            notes="Weighted Training for Holidays"
        )

      
if __name__ == "__main__":
    main()