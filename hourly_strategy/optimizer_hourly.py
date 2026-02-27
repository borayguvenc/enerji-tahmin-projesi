
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

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# src is locally copied to p0/hourly_strategy/src
sys.path.append(os.path.join(current_dir, 'src'))

# --- IMPORTS ---
from src.data_manager import DataManager
from config import RAW_TARGET_COL 

# --- GLOBAL SETTINGS ---
N_TRIALS = 20 # Number of trials per model per hour (Adjustable)
HOURS = range(24)
OUTPUT_FILE = "hourly_params.json"

# --- HELPER FUNCTIONS ---
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def get_tscv_splits(X, y, n_splits=3, test_size=30): # Small test size for specific hour data
    """
    Returns splits for TimeSeriesSplit on the filtered hourly data.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        splits.append((X_train, y_train, X_test, y_test))
    return splits

# ============================================================
# OBJECTIVE FUNCTIONS
# ============================================================

def objective_xgb(trial, X, y):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'n_jobs': -1,
        'n_estimators': 1000, # Lower than global to speed up
        'early_stopping_rounds': 50,
        'enable_categorical': True,
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }

    scores = []
    splits = get_tscv_splits(X, y)
    
    for X_train, y_train, X_test, y_test in splits:
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

def objective_lgbm(trial, X, y):
    param = {
        'objective': 'regression',
        'metric': 'mae',
        'n_jobs': -1,
        'n_estimators': 1000,
        'verbosity': -1,
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }

    scores = []
    splits = get_tscv_splits(X, y)

    for X_train, y_train, X_test, y_test in splits:
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
        preds = model.predict(X_test)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

def objective_cat(trial, X, y, cat_features):
    param = {
        'loss_function': 'MAE',
        'eval_metric': 'MAPE',
        'iterations': 1000,
        'verbose': False,
        'early_stopping_rounds': 50,
        'allow_writing_files': False,
        'cat_features': cat_features,
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 5),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20)
    }

    scores = []
    splits = get_tscv_splits(X, y)

    for X_train, y_train, X_test, y_test in splits:
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)
        
        model = CatBoostRegressor(**param)
        model.fit(train_pool, eval_set=test_pool)
        preds = model.predict(test_pool)
        scores.append(calculate_mape(y_test, preds))

    return np.mean(scores)

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("⏳ Loading Data for Optimization...")
    dm = DataManager()
    dm.load_and_preprocess()
    
    # Feature/Target Separation
    target_col = RAW_TARGET_COL
    feature_cols = [c for c in dm.data.columns if c not in [target_col, 'Tarih', 'Datetime']]
    X_global = dm.data[feature_cols]
    y_global = dm.data[target_col]
    
    # Ensure 'Saat' column exists
    if 'Saat' not in X_global.columns:
         if isinstance(X_global.index, pd.DatetimeIndex):
             X_global = X_global.copy()
             X_global['Saat'] = X_global.index.hour
         else:
             raise ValueError("'Saat' column missing and index is not DatetimeIndex")
             
    cat_features_names = X_global.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Dictionary to store results
    hourly_params = {}
    
    # Load existing if available (to resume)
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                hourly_params = json.load(f)
            # Convert keys back to int (JSON keys are strings)
            hourly_params = {int(k): v for k, v in hourly_params.items()}
            print(f"🔄 Resuming from existing {OUTPUT_FILE} with {len(hourly_params)} hours completed.")
        except Exception as e:
            print(f"⚠️ Could not load existing params: {e}")

    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce output
    
    print(f"🚀 Starting Hourly Optimization ({N_TRIALS} trials per model)...")
    
    for h in HOURS:
        if h in hourly_params and len(hourly_params[h]) == 3:
            print(f"✅ Hour {h} already completed. Skipping.")
            continue
            
        print(f"\n>>> 🔍 Optimizing Hour: {h:02d} <<<")
        
        # Filter Data
        mask = (X_global['Saat'] == h)
        X_h = X_global[mask].copy()
        y_h = y_global[mask].copy()
        
        if 'Saat' in X_h.columns:
            X_h.drop(columns=['Saat'], inplace=True)
            
        if len(X_h) < 100:
            print(f"⚠️ Not enough data for Hour {h}. Skipping.")
            continue

        if h not in hourly_params:
            hourly_params[h] = {}

        # 1. Optimize XGB
        if 'xgb' not in hourly_params[h]:
            print(f"   - Tuning XGBoost...")
            study_xgb = optuna.create_study(direction='minimize')
            study_xgb.optimize(lambda trial: objective_xgb(trial, X_h, y_h), n_trials=N_TRIALS)
            hourly_params[h]['xgb'] = study_xgb.best_params
            print(f"     -> Best MAPE: {study_xgb.best_value:.4f}%")
        else:
            print(f"   - XGBoost already done.")
            
        # 2. Optimize LGBM
        if 'lgbm' not in hourly_params[h]:
            print(f"   - Tuning LightGBM...")
            study_lgbm = optuna.create_study(direction='minimize')
            study_lgbm.optimize(lambda trial: objective_lgbm(trial, X_h, y_h), n_trials=N_TRIALS)
            hourly_params[h]['lgbm'] = study_lgbm.best_params
            print(f"     -> Best MAPE: {study_lgbm.best_value:.4f}%")
        else:
             print(f"   - LightGBM already done.")

        # 3. Optimize CAT
        if 'cat' not in hourly_params[h]:
            print(f"   - Tuning CatBoost...")
            study_cat = optuna.create_study(direction='minimize')
            study_cat.optimize(lambda trial: objective_cat(trial, X_h, y_h, cat_features_names), n_trials=N_TRIALS)
            hourly_params[h]['cat'] = study_cat.best_params
            print(f"     -> Best MAPE: {study_cat.best_value:.4f}%")
        else:
             print(f"   - CatBoost already done.")
            
        # SAVE PROGRESS
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(hourly_params, f, indent=4)
        print(f"💾 Progress saved for Hour {h}.")
        
    print(f"\n✅ All Hourly Optimizations Complete! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
