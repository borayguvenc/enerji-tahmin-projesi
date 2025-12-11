import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
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

# --- DATA YÜKLEME (GLOBAL) ---
# Optuna her trial'da veriyi tekrar yüklemesin diye en başta bir kere yüklüyoruz.
print("Veri yükleniyor...")
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

# ============================================================
# 1. XGBOOST OBJECTIVE
# ============================================================
def objective_xgb(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_jobs': -1,
        'n_estimators': 2000, # Optuna early stopping ile durduracak
        'early_stopping_rounds': 50,
        'enable_categorical': True,
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0)
    }

    tscv = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE) # Hız için split 3'e düşürülebilir
    cv_scores = []

    for train_index, test_index in tscv.split(X_global):
        X_train, X_test = X_global.iloc[train_index], X_global.iloc[test_index]
        y_train, y_test = y_global.iloc[train_index], y_global.iloc[test_index]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        preds = model.predict(X_test)
        cv_scores.append(calculate_mape(y_test, preds))

    return np.mean(cv_scores)

# ============================================================
# 2. LIGHTGBM OBJECTIVE
# ============================================================
def objective_lgbm(trial):
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'n_jobs': -1,
        'n_estimators': 2000,
        'verbosity': -1,
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100), # LGBM için en kritik parametre
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0)
    }

    tscv = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE)
    cv_scores = []

    for train_index, test_index in tscv.split(X_global):
        X_train, X_test = X_global.iloc[train_index], X_global.iloc[test_index]
        y_train, y_test = y_global.iloc[train_index], y_global.iloc[test_index]
        
        # LightGBM için Early Stopping Callback
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        
        preds = model.predict(X_test)
        cv_scores.append(calculate_mape(y_test, preds))

    return np.mean(cv_scores)

# ============================================================
# 3. CATBOOST OBJECTIVE
# ============================================================
def objective_cat(trial):
    param = {
        'loss_function': 'MAE',
        'eval_metric': 'MAPE',
        'iterations': 2000,
        'verbose': False,
        'early_stopping_rounds': 50,
        'allow_writing_files': False,
        'cat_features': cat_features_names, # Otomatik tespit edilen kategorik sütunlar
        
        # Optimize Edilecekler
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }

    tscv = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE)
    cv_scores = []

    for train_index, test_index in tscv.split(X_global):
        X_train, X_test = X_global.iloc[train_index], X_global.iloc[test_index]
        y_train, y_test = y_global.iloc[train_index], y_global.iloc[test_index]
        
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        preds = model.predict(X_test)
        cv_scores.append(calculate_mape(y_test, preds))

    return np.mean(cv_scores)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='XGB', help='Model: XGB, LGBM, CAT')
    parser.add_argument('--trials', type=int, default=50, help='Deneme sayısı')
    args = parser.parse_args()
    
    model_name = args.model.upper()
    n_trials = args.trials
    
    print(f"\n🚀 {model_name} İçin Optimizasyon Başlıyor ({n_trials} Trial)...")
    
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
    print(f"En İyi Skor (MAPE): %{study.best_value:.4f}")
    print("En İyi Parametreler:")
    print(study.best_params)
    
    # Parametreleri JSON olarak kaydet
    file_name = f"best_params_{model_name.lower()}.json"
    with open(file_name, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\n📁 Parametreler '{file_name}' dosyasına kaydedildi.")