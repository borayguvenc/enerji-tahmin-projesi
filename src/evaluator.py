import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import sys
import os

# Gerekli Yöneticileri İçe Aktar
from src.model_manager import ModelManager       # XGBoost
from src.lightgbm_manager import LightGBMManager # LightGBM
from src.catboost_manager import CatBoostManager # CatBoost
from src.catboost_bayram_manager import CatBoostBayramManager # CatBoost Sniper

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
        Hem skorları hem de birleştirilmiş tüm tahminleri döndürür.
        Return: (cv_scores, full_predictions_df)
        """
        print(f"\n [Evaluator] Cross Validation Başlıyor ({self.n_splits} Fold)...")
        print(f" Model Tipi: {model_type}")
        
        cv_scores = []
        fold = 1

        # KUMBARA: Tüm tahminleri burada biriktireceğiz
        storage = {
            'Date': [],
            'Actual': [],
            'XGB_Pred': [],
            'LGBM_Pred': [],
            'CAT_Pred': [],
            'Ensemble_Pred': []
        }

        for train_index, test_index in self.tscv.split(X):
            # 1. Veriyi Böl
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            test_start = X_test.index.min()
            test_end = X_test.index.max()
            print(f" Fold {fold}/{self.n_splits} | Test Dönemi: {test_start} -> {test_end}")

            # Tarihleri ve Gerçek Değerleri Sakla
            storage['Date'].extend(X_test.index)
            storage['Actual'].extend(y_test.values)

            # Geçici değişkenler (Eğer model çalışmazsa None kalmasın diye)
            p_xgb = np.zeros(len(y_test))
            p_lgbm = np.zeros(len(y_test))
            p_cat = np.zeros(len(y_test))

            # --- MODEL EĞİTİM VE TAHMİN ---
            
            # Eğer Mod 'ALL' ise hepsini eğit
            if model_type == 'ALL':
                # XGBoost
                mm = ModelManager()
                mm.train_model(X_train, y_train, X_test, y_test)
                p_xgb = mm.model.predict(X_test)

                # LightGBM
                lgbm = LightGBMManager()
                lgbm.train_model(X_train, y_train, X_test, y_test)
                p_lgbm = lgbm.model.predict(X_test)

                # CatBoost
                cat = CatBoostManager()
                cat.train_model(X_train, y_train, X_test, y_test)
                p_cat = cat.model.predict(X_test)

                # Ensemble
                p_ensemble = (p_xgb + p_lgbm + p_cat) / 3
                
                # Kumbaraya At
                storage['XGB_Pred'].extend(p_xgb)
                storage['LGBM_Pred'].extend(p_lgbm)
                storage['CAT_Pred'].extend(p_cat)
                storage['Ensemble_Pred'].extend(p_ensemble)

                final_preds_for_score = p_ensemble

            # Eğer Tekli Mod ise (Örn: Sadece XGB)
            else:
                if model_type == 'XGB':
                    man = ModelManager()
                    man.train_model(X_train, y_train, X_test, y_test)
                    preds = man.model.predict(X_test)
                elif model_type == 'LGBM':
                    man = LightGBMManager()
                    man.train_model(X_train, y_train, X_test, y_test)
                    preds = man.model.predict(X_test)
                elif model_type == 'CAT':
                    man = CatBoostManager()
                    man.train_model(X_train, y_train, X_test, y_test)
                    preds = man.model.predict(X_test)
                elif model_type == 'SNIPER':
                    man = CatBoostBayramManager()
                    man.train_model(X_train, y_train, X_test, y_test)
                    preds = man.model.predict(X_test)
                
                final_preds_for_score = preds
                
                # Tekli modda diğerlerini boş geçiyoruz (veya aynısını yazıyoruz)
                # Mantık hatası olmasın diye tekli modda sadece ensemble kolonuna yazıyorum
                storage['Ensemble_Pred'].extend(preds)

            # Skor Hesapla
            mape = calculate_mape(y_test, final_preds_for_score)
            cv_scores.append(mape)
            
            print(f"  Fold {fold} MAPE: %{mape:.2f}")
            print("  --------------------------------------------------")
            fold += 1

        # Döngü Bitti: Kumbarayı DataFrame'e çevir
        # Eğer tekli moddaysa diğer sütunlar boş kalabilir, sorun değil.
        full_df = pd.DataFrame({
            'Actual': storage['Actual'],
            'Ensemble_Pred': storage['Ensemble_Pred']
        }, index=storage['Date'])

        if model_type == 'ALL':
            full_df['XGB_Pred'] = storage['XGB_Pred']
            full_df['LGBM_Pred'] = storage['LGBM_Pred']
            full_df['CAT_Pred'] = storage['CAT_Pred']

        return cv_scores, full_df

    def print_summary(self, scores):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print("\n📊 === CROSS VALIDATION SONUÇ RAPORU ===")
        print(f"Tüm Skorlar (MAPE): {['%.2f%%' % s for s in scores]}")
        print(f"Ortalama MAPE     : %{mean_score:.2f}")
        print(f"Standart Sapma    : {std_score:.2f} ")
        print("========================================\n")