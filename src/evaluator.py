import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

# Gerekli Yöneticileri İçe Aktar
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class Evaluator:
    def __init__(self, n_splits=5, test_size=720):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    def run_cross_validation(self, X, y, model_type='XGB'):
        """
        model_type: 'XGB', 'LGBM', 'CAT', 'ALL'
        """
        print(f"\n [Evaluator] Cross Validation Başlıyor ({self.n_splits} Fold)...")
        print(f" Model Tipi: {model_type}")
        
        cv_scores = []
        fold = 1

        for train_index, test_index in self.tscv.split(X):
            # 1. Veriyi Böl
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            test_start = X_test.index.min()
            test_end = X_test.index.max()
            print(f" Fold {fold}/{self.n_splits} | Test Dönemi: {test_start} -> {test_end}")

            # -------------------------------------------------
            # ENSEMBLE MODU (HER FOLD İÇİN 3 MODEL EĞİT)
            # -------------------------------------------------
            if model_type == 'ALL':
                # 1. XGBoost
                mm = ModelManager()
                mm.train_model(X_train, y_train, X_test, y_test)
                p1 = mm.model.predict(X_test)
                
                # 2. LightGBM
                lgbm = LightGBMManager()
                lgbm.train_model(X_train, y_train, X_test, y_test)
                p2 = lgbm.model.predict(X_test)
                
                # 3. CatBoost
                cat = CatBoostManager()
                cat.train_model(X_train, y_train, X_test, y_test)
                p3 = cat.model.predict(X_test)
                
                # Ortalama Al
                preds = (p1 + p2 + p3) / 3
                
            # -------------------------------------------------
            # TEKLİ MODLAR
            # -------------------------------------------------
            else:
                if model_type == 'XGB':
                    manager = ModelManager()
                elif model_type == 'LGBM':
                    manager = LightGBMManager()
                elif model_type == 'CAT':
                    manager = CatBoostManager()
                else:
                    raise ValueError(f"Geçersiz model tipi: {model_type}")
                
                manager.train_model(X_train, y_train, X_test, y_test)
                preds = manager.model.predict(X_test)
            
            # 5. Skor Hesapla
            mape = calculate_mape(y_test, preds)
            cv_scores.append(mape)
            
            print(f"  Fold {fold} MAPE: %{mape:.2f}")
            print("  --------------------------------------------------")
            fold += 1

        return cv_scores

    def print_summary(self, scores):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print("\n📊 === CROSS VALIDATION SONUÇ RAPORU ===")
        print(f"Tüm Skorlar (MAPE): {['%.2f%%' % s for s in scores]}")
        print(f"Ortalama MAPE     : %{mean_score:.2f}")
        print(f"Standart Sapma    : {std_score:.2f} ")
        print("========================================\n")