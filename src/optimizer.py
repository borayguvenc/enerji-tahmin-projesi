import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import sys
import os
import json
import argparse
from sklearn.model_selection import TimeSeriesSplit 

# --- YOL AYARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- İMPORTLAR ---
from src.data_manager import DataManager
from config import TEST_SIZE, RAW_TARGET_COL 

# --- GLOBAL DEĞİŞKENLER ---
# Bunlar komut satırından gelen argümanlarla doldurulacak
TARGET_START_DATE = None
TARGET_END_DATE = None

# --- DATA YÜKLEME (GLOBAL) ---
print("⏳ Veri yükleniyor ve ön işleme yapılıyor...")
dm = DataManager()
dm.load_and_preprocess()

# Feature ve Target Ayrımı
target_col = RAW_TARGET_COL
feature_cols = [c for c in dm.data.columns if c not in [target_col, 'Tarih', 'Datetime']]
X_global = dm.data[feature_cols]
y_global = dm.data[target_col]

# CatBoost için kategorik sütunları bulalım
cat_features_names = X_global.select_dtypes(include=['object', 'category']).columns.tolist()

# --- YARDIMCI FONKSİYONLAR ---
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def get_train_test_data():
    """
    Eğer tarih verilmişse o tarihe göre, verilmemişse TimeSeriesSplit mantığına göre
    tek bir Train/Test çifti döndürür (Optimizasyon hızı için).
    """
    if TARGET_START_DATE and TARGET_END_DATE:
        # --- SENARYO A: HEDEF ODAKLI (BELİRLİ BİR AY) ---
        # Test seti: Belirtilen tarih aralığı
        # Train seti: O tarihten önceki tüm veri
        mask_test = (X_global.index >= TARGET_START_DATE) & (X_global.index <= TARGET_END_DATE)
        mask_train = (X_global.index < TARGET_START_DATE)
        
        X_train = X_global[mask_train]
        y_train = y_global[mask_train]
        X_test = X_global[mask_test]
        y_test = y_global[mask_test]
        
        return [(X_train, y_train, X_test, y_test)]
    
    else:
        # --- SENARYO B: GENEL (SON DÖNEM) ---
        # TimeSeriesSplit kullanır ama sadece son split'i döndürürüz ki
        # Optuna çok yavaşlamasın (veya istersen döngüye sokabilirsin).
        tscv = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE)
        splits = []
        for train_index, test_index in tscv.split(X_global):
            X_train, X_test = X_global.iloc[train_index], X_global.iloc[test_index]
            y_train, y_test = y_global.iloc[train_index], y_global.iloc[test_index]
            splits.append((X_train, y_train, X_test, y_test))
        
        # Son 2 split'i kullan (Ortalama almak için)
        return splits[-2:] 

# ============================================================
# 1. XGBOOST OBJECTIVE
# ============================================================
def objective_xgb(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_jobs': -1,
        'n_estimators': 3000, 
        'early_stopping_rounds': 100,
        'enable_categorical': True,
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
    }

    scores = []
    splits = get_train_test_data()
    
    for X_train, y_train, X_test, y_test in splits:
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

# ============================================================
# 2. LIGHTGBM OBJECTIVE
# ============================================================
def objective_lgbm(trial):
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'n_jobs': -1,
        'n_estimators': 3000,
        'verbosity': -1,
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
    }

    scores = []
    splits = get_train_test_data()

    for X_train, y_train, X_test, y_test in splits:
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        preds = model.predict(X_test)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

# ============================================================
# 3. CATBOOST OBJECTIVE
# ============================================================
def objective_cat(trial):
    param = {
        'loss_function': 'MAE',
        'eval_metric': 'MAPE',
        'iterations': 3000,
        'verbose': False,
        'early_stopping_rounds': 100,
        'allow_writing_files': False,
        'cat_features': cat_features_names,
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 15),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50)
    }

    scores = []
    splits = get_train_test_data()

    for X_train, y_train, X_test, y_test in splits:
        # CatBoost Pool (Hız için)
        train_pool = Pool(X_train, y_train, cat_features=cat_features_names)
        test_pool = Pool(X_test, y_test, cat_features=cat_features_names)
        
        model = CatBoostRegressor(**param)
        model.fit(train_pool, eval_set=test_pool)
        preds = model.predict(test_pool)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='XGB', help='Model: XGB, LGBM, CAT')
    parser.add_argument('--trials', type=int, default=50, help='Deneme sayısı')
    # YENİ EKLENEN PARAMETRELER
    parser.add_argument('--start_date', type=str, default=None, help='Hedef Ay Başlangıcı (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='Hedef Ay Bitişi (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    model_name = args.model.upper()
    n_trials = args.trials
    
    # Global değişkenlere ata
    if args.start_date and args.end_date:
        TARGET_START_DATE = args.start_date
        TARGET_END_DATE = args.end_date
        print(f"\n🎯 HEDEF ODAKLI OPTİMİZASYON: {TARGET_START_DATE} ile {TARGET_END_DATE} arası.")
        print("   (Model bu tarih aralığındaki hatayı minimize etmeye çalışacak)")
    else:
        print(f"\n🔄 GENEL OPTİMİZASYON (Son dönemler baz alınıyor)")
    
    print(f"🚀 {model_name} İçin Optimizasyon Başlıyor ({n_trials} Trial)...")
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='minimize')
    
    if model_name == 'XGB':
        study.optimize(objective_xgb, n_trials=n_trials)
    elif model_name == 'LGBM':
        study.optimize(objective_lgbm, n_trials=n_trials)
    elif model_name == 'CAT':
        study.optimize(objective_cat, n_trials=n_trials)
    else:
        print("Geçersiz model ismi! (XGB, LGBM, CAT)")
        sys.exit()
        
    print("\n✅ OPTİMİZASYON TAMAMLANDI!")
    print(f"🏆 En İyi Skor (MAPE): %{study.best_value:.4f}")
    print("💎 En İyi Parametreler:")
    print(json.dumps(study.best_params, indent=4))
    
    # Dosya ismine tarih ekleyelim ki karışmasın
    suffix = f"_{args.start_date}_{args.end_date}" if args.start_date else "_general"
    file_name = f"best_params_{model_name.lower()}{suffix}.json"
    
    with open(file_name, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n📁 Parametreler '{file_name}' dosyasına kaydedildi.")