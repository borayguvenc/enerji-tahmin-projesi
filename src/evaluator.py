import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

# Gerekli Yöneticileri İçe Aktar
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost
from src.catboost_manager_tuned import CatBoostManagerTuned # CatBoost Tuned
from src.ann_manager import ANNManager

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

    def run_cross_validation(self, X, y, model_type='CAT', mode='actual'):
        """
        Hem skorları hem de birleştirilmiş tüm tahminleri döndürür.
        mode: 'fc' (Forecast takaslı) veya 'actual' (Sadece gerçek verilerle)
        """
        print(f"\n [Evaluator] Cross Validation Başlıyor ({self.n_splits} Fold)...")
        print(f" Model Tipi: {model_type} | Çalışma Modu: {mode}")
        
        cv_scores = []
        fold = 1

        storage = {
            'Date': [],
            'Actual': [],
            'XGB_Pred': [],
            'LGBM_Pred': [],
            'CAT_Pred': [],
            'Ensemble_Pred': [],
            'fold_id': []
        }

        for train_index, test_index in self.tscv.split(X):
            # 1. Ham Veriyi Böl
            X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 2. MODA GÖRE VERİ TAKASI
            if mode == 'fc':
                fc_cols = [c for c in X.columns if c.endswith('_fc')]
                X_train = X_train.drop(columns=fc_cols)
                for f_col in fc_cols:
                    actual_col = f_col.replace('_fc', '_actual')
                    if actual_col in X_test.columns:
                        X_test[actual_col] = X_test[f_col]
                X_test = X_test.drop(columns=fc_cols)
                print(f"  Fold {fold}: Forecast sütunları actual sütunlarla değiştirildi.")

            elif mode == 'actual':
                fc_cols = [c for c in X.columns if c.endswith('_fc')]
                X_train = X_train.drop(columns=fc_cols)
                X_test = X_test.drop(columns=fc_cols)
                print(f"  Fold {fold}: Tüm forecast sütunları kaldırıldı.")
            
            # Tarihleri ve Gerçek Değerleri Sakla
            storage['Date'].extend(X_test.index)
            storage['Actual'].extend(y_test.values)

            test_start = X_test.index.min()
            test_end = X_test.index.max()
            print(f" Fold {fold}/{self.n_splits} | Test Dönemi: {test_start} -> {test_end}")

            # 3. MODEL EĞİTİM VE TAHMİN
            p_xgb = np.zeros(len(y_test))
            p_lgbm = np.zeros(len(y_test))
            p_cat = np.zeros(len(y_test))

            if model_type == 'ALL':
                mm = ModelManager()
                mm.train_model(X_train, y_train, X_test, y_test)
                p_xgb = mm.model.predict(X_test)

                lgbm = LightGBMManager()
                lgbm.train_model(X_train, y_train, X_test, y_test)
                p_lgbm = lgbm.model.predict(X_test)

                cat = CatBoostManager()
                cat.train_model(X_train, y_train, X_test, y_test)
                p_cat = cat.model.predict(X_test)

                p_ensemble = (p_xgb + p_lgbm + p_cat) / 3
                
                storage['XGB_Pred'].extend(p_xgb)
                storage['LGBM_Pred'].extend(p_lgbm)
                storage['CAT_Pred'].extend(p_cat)
                storage['Ensemble_Pred'].extend(p_ensemble)
                storage['fold_id'].extend([fold] * len(y_test))
                final_preds_for_score = p_ensemble
            else:
                if model_type == 'XGB': man = ModelManager()
                elif model_type == 'LGBM': man = LightGBMManager()
                elif model_type == 'CAT': man = CatBoostManager()
                elif model_type == 'CAT_TUNED': man = CatBoostManagerTuned()
                elif model_type == 'ANN': man = ANNManager()
                
                man.train_model(X_train, y_train, X_test, y_test)
                if model_type == 'ANN':
                    preds = man.predict(X_test)
                else:
                    preds = man.model.predict(X_test)
                storage['Ensemble_Pred'].extend(preds)
                storage['fold_id'].extend([fold] * len(y_test))
                final_preds_for_score = preds

            mape = calculate_mape(y_test, final_preds_for_score)
            cv_scores.append(mape)
            print(f"  Fold {fold} MAPE: %{mape:.2f}")
            print("  --------------------------------------------------")
            fold += 1

        full_df = pd.DataFrame({
            'Actual': storage['Actual'],
            'Ensemble_Pred': storage['Ensemble_Pred']
        }, index=storage['Date'])

        if model_type == 'ALL':
            full_df['XGB_Pred'] = storage['XGB_Pred']
            full_df['LGBM_Pred'] = storage['LGBM_Pred']
            full_df['CAT_Pred'] = storage['CAT_Pred']
            full_df['fold_id'] = storage['fold_id']

        return cv_scores, full_df

    # --- ÖNEMLİ: BU METODUN SINIFIN EN DIŞINDA OLMADIĞINDAN EMİN OL ---
    def print_summary(self, scores):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print("\n📊 === CROSS VALIDATION SONUÇ RAPORU ===")
        print(f"Tüm Skorlar (MAPE): {['%.2f%%' % s for s in scores]}")
        print(f"Ortalama MAPE     : %{mean_score:.2f}")
        print(f"Standart Sapma    : {std_score:.2f} ")
        print("========================================\n")