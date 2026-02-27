
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Add parent directory to path to import original src if needed, 
# but we copied src locally so we import from local src.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Managers from local src
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost

class HourlyGrandEnsemble:
    def __init__(self, project_root):
        self.project_root = project_root
        self.models = {} # Structure: { hour: { 'xgb': model, 'lgbm': model, 'cat': model, 'meta': model } }
        self.hours = range(24)
        
        # Load optimized params
        self.params_path = os.path.join(project_root, 'hourly_params.json')
        self.optimized_params = {}
        if os.path.exists(self.params_path):
            try:
                import json
                with open(self.params_path, 'r') as f:
                    self.optimized_params = json.load(f)
                print(f"[HourlyGrandEnsemble] Loaded optimized parameters from {self.params_path}")
            except Exception as e:
                print(f"[HourlyGrandEnsemble] Failed to load params: {e}")
        else:
            print(f"[HourlyGrandEnsemble] No optimized params found at {self.params_path}. Using defaults.")
        
    def fit(self, X, y):
        """
        Trains 24 separate Grand Ensembles (one for each hour).
        For each hour:
          1. Filter data for that hour.
          2. Perform Internal CV to generate OOF predictions for Stacking.
          3. Train Meta-Model (Linear Regression) on OOF preds.
          4. Retrain Base Models on full data for that hour.
        """
        print(f"\n[HourlyGrandEnsemble] Starting training for 24 hours...")
        
        # Ensure 'Saat' is available in X for filtering
        if 'Saat' not in X.columns:
            print(f"[DEBUG] 'Saat' not found in columns: {X.columns}")
            # Try to recover from index if it's datetime
            if isinstance(X.index, pd.DatetimeIndex):
                 print("[DEBUG] 'Saat' recovered from DatetimeIndex")
                 X = X.copy()
                 X['Saat'] = X.index.hour
            else:
                raise ValueError("Input X must have 'Saat' column or DatetimeIndex to filter by hour.")
        
        print(f"[DEBUG] X shape: {X.shape}")
        print(f"[DEBUG] X['Saat'] unique values: {X['Saat'].unique()}")

        for h in self.hours:
            print(f"\n>>> 🕒 Training Hour: {h:02d} <<<")
            
            # 1. Filter Data for Hour h
            mask = (X['Saat'] == h)
            X_h = X[mask].copy()
            y_h = y[mask].copy()
            
            # Drop 'Saat' column as it's constant for this model
            if 'Saat' in X_h.columns:
                X_h.drop(columns=['Saat'], inplace=True)
                
            # Dictionary to store this hour's models
            self.models[h] = {}
            
            # --- STACKING: Generate OOF Predictions via CV ---
            # We need OOF preds to train the LinearRegression Meta-Model without leakage.
            # Using 5 splits for internal CV to be reasonably fast.
            tscv = TimeSeriesSplit(n_splits=5, test_size=30) 
            
            oof_preds = pd.DataFrame(index=X_h.index)
            oof_preds['Actual'] = y_h
            oof_preds['XGB'] = np.nan
            oof_preds['LGBM'] = np.nan
            oof_preds['CAT'] = np.nan
            
            print(f"   -> Generating OOF predictions for Stacking (Internal CV)...")
            
            # Loop over folds
            fold_idx = 0
            for train_idx, val_idx in tscv.split(X_h):
                fold_idx += 1
                # Split
                X_tr, X_val = X_h.iloc[train_idx], X_h.iloc[val_idx]
                y_tr, y_val = y_h.iloc[train_idx], y_h.iloc[val_idx]
                
                # Train Temp Models
                # Train Temp Models
                # XGB
                xgb_temp = ModelManager() 
                # Use train_model to initialize and fit
                xgb_temp.train_model(X_tr, y_tr, X_val, y_val)
                p_xgb = xgb_temp.model.predict(X_val)
                
                # LGBM
                lgbm_temp = LightGBMManager()
                lgbm_temp.train_model(X_tr, y_tr, X_val, y_val)
                p_lgbm = lgbm_temp.model.predict(X_val)
                
                # CAT
                cat_temp = CatBoostManager()
                cat_temp.train_model(X_tr, y_tr, X_val, y_val)
                p_cat = cat_temp.model.predict(X_val)
                
                # Store OOF
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc('XGB')] = p_xgb
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc('LGBM')] = p_lgbm
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc('CAT')] = p_cat
            
            # Drop NaNs (first part of data used for training won't have OOF preds)
            oof_clean = oof_preds.dropna()
            
            if len(oof_clean) == 0:
                print("   -> WARNING: Not enough data for stacking! Using simple average.")
                self.models[h]['meta'] = None
            else:
                # Train Meta Model
                meta_features = oof_clean[['XGB', 'LGBM', 'CAT']]
                meta_target = oof_clean['Actual']
                
                meta_model = LinearRegression()
                meta_model.fit(meta_features, meta_target)
                
                self.models[h]['meta'] = meta_model
                print(f"   -> Meta-Model Coefficients: {meta_model.coef_}")
                
            
            # Load optimized params if available
            hour_params = {}
            if self.optimized_params and str(h) in self.optimized_params:
                hour_params = self.optimized_params[str(h)]
            elif self.optimized_params and h in self.optimized_params:
                hour_params = self.optimized_params[h]

            # --- FINAL TRAINING: Train Base Models on FULL Hour Data ---
            print(f"   -> Retraining base models on full data for Hour {h}...")
            
            # XGB
            final_xgb = ModelManager()
            # Split for early stopping (simple 90/10 split)
            split_point = int(len(X_h) * 0.9)
            X_fin_tr, X_fin_val = X_h.iloc[:split_point], X_h.iloc[split_point:]
            y_fin_tr, y_fin_val = y_h.iloc[:split_point], y_h.iloc[split_point:]
            
            xgb_params = hour_params.get('xgb')
            final_xgb.train_model(X_fin_tr, y_fin_tr, X_fin_val, y_fin_val, params=xgb_params)
            self.models[h]['xgb'] = final_xgb.model
            
            # LGBM
            final_lgbm = LightGBMManager()
            lgbm_params = hour_params.get('lgbm')
            final_lgbm.train_model(X_fin_tr, y_fin_tr, X_fin_val, y_fin_val, params=lgbm_params)
            self.models[h]['lgbm'] = final_lgbm.model
            
            # CAT
            final_cat = CatBoostManager()
            cat_params = hour_params.get('cat')
            final_cat.train_model(X_fin_tr, y_fin_tr, X_fin_val, y_fin_val, params=cat_params)
            self.models[h]['cat'] = final_cat.model
            
            print(f"   -> Hour {h} Training Complete.")

    def predict(self, X):
        """
        Predicts using the 24 hourly models.
        """
        print(f"\n[HourlyGrandEnsemble] Predicting...")
        
        # Prepare Result Container
        # We need to preserve the index and order
        results = pd.Series(index=X.index, dtype=float, name='Prediction')
        results[:] = np.nan # Initialize with NaN
        
        if 'Saat' not in X.columns:
             if isinstance(X.index, pd.DatetimeIndex):
                 X = X.copy()
                 X['Saat'] = X.index.hour
        
        for h in self.hours:
            # Filter rows for this hour
            mask = (X['Saat'] == h)
            if not mask.any():
                continue
                
            X_h = X[mask].copy()
            # Drop Saat
            if 'Saat' in X_h.columns:
                 X_h.drop(columns=['Saat'], inplace=True)
            
            # Get Models
            if h not in self.models:
                print(f"Warning: No model found for hour {h}")
                continue
                
            model_xgb = self.models[h]['xgb']
            model_lgbm = self.models[h]['lgbm']
            model_cat = self.models[h]['cat']
            model_meta = self.models[h].get('meta')
            
            # Base Predictions
            p_xgb = model_xgb.predict(X_h)
            p_lgbm = model_lgbm.predict(X_h)
            p_cat = model_cat.predict(X_h)
            
            # Stack
            if model_meta:
                meta_input = pd.DataFrame({'XGB': p_xgb, 'LGBM': p_lgbm, 'CAT': p_cat})
                p_final = model_meta.predict(meta_input)
            else:
                p_final = (p_xgb + p_lgbm + p_cat) / 3.0
            
            # Place in results
            results.loc[mask] = p_final
            
        return results

