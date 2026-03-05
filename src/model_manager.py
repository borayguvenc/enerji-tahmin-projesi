import xgboost as xgb
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --- CONFIG YOLU AYARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config import (MODEL_NAME)

def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) hesaplar.
    """
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.join(project_root, 'Model')
        # Model klasörü yoksa oluştur
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train, y_train, X_test, y_test):
        print("[ModelManager] Initializing XGBoost Regressor...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            
            # --- OPTIMIZED PARAMS ---
            learning_rate=0.09931882542569642,
            max_depth=4,
            subsample=0.7828662239525903,
            colsample_bytree=0.574383187490635,
            min_child_weight=13,              # Eklendi
            reg_alpha=8.924757905144489,    # Eklendi (L1 Regularization)
            reg_lambda=2.005839273119691,  # Eklendi (L2 Regularization)
            # ------------------------
            
            objective='reg:squarederror', 
            eval_metric='mae',  
            enable_categorical=True, 
            early_stopping_rounds=50,
            n_jobs=-1
        )

        print(f"[ModelManager] Training started on {len(X_train)} samples...")

        """ 
        # --- ADIM 2: CEZALARI ARTIR (Katsayılar) ---
        weights = np.ones(len(X_train))
        # A. Ramazan Günleri (Örn: 3 Kat Önemli)
        if 'Is_Ramadan' in X_train.columns:
            # Sütunu bul ve maske oluştur
            mask = (X_train['Is_Ramadan'] == 1).values
            weights[mask] *= 3.0
            
        # B. Sahur Saatleri (Örn: 5 Kat Önemli - Gece hatalarını düzeltmek için)
        if 'Is_Sahur' in X_train.columns:
            mask = (X_train['Is_Sahur'] == 1).values
            weights[mask] *= 5.0
            
        # C. Milli/Dini Bayramlar (Örn: 10 Kat Önemli - Mart sonundaki çukuru düzeltmek için)
        if 'Milli_Bayram' in X_train.columns: # Sütun adın 'Is_Holiday' ise onu yaz
            mask = (X_train['Milli_Bayram'] == 1).values
            weights[mask] *= 10.0
            
        if 'Kurban_Bayram' in X_train.columns:
            mask = (X_train['Kurban_Bayram'] == 1).values
            weights[mask] *= 10.0

        # D. Mart Soğukları (Eğer data_manager'da eklediysen)
        if 'March_Heating_Degree' in X_train.columns:
            # Soğuk varsa (Değer 0'dan büyükse) ağırlığı artır
            mask = (X_train['March_Heating_Degree'] > 0).values
            weights[mask] *= 2.0

        """
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            #sample_weight=weights,
            verbose=100
        )
        print("[ModelManager] Training finished.")

    def evaluate(self, X_test, y_test):
        if self.model is None:
            print("Model is not trained yet!")
            return

        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Artık dışarıdaki fonksiyonu rahatça çağırabiliriz
        mape = calculate_mape(y_test, preds)
        
        print("\n--- Model Performance on Test Set ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : {mape:.2f}%") 
        print("-------------------------------------\n")
        
        return preds

    def save_model(self, filename=MODEL_NAME):
        if self.model is None:
            print("No model to save.")
            return

        save_path = os.path.join(self.model_dir, filename)
        self.model.save_model(save_path)
        print(f"[ModelManager] Model saved to: {save_path}")

    def load_model(self, filename=MODEL_NAME):
        load_path = os.path.join(self.model_dir, filename)
        if os.path.exists(load_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(load_path)
            print(f"[ModelManager] Model loaded from: {load_path}")
        else:
            print(f"Model file not found at: {load_path}")

    def get_feature_importance(self, max_num=20):
        """
        XGBoost Gain skorlarını kullanarak önem derecelerini raporlar.
        """
        if self.model is None:
            print("Model henüz eğitilmedi!")
            return None
        
        # 'gain' parametresi hatayı ne kadar azalttığını gösterir (En kritiği budur)
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        # Görselleştirme
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model, importance_type='gain', max_num_features=max_num, 
                           height=0.5, title='XGBoost Feature Importance (Gain)')
        plt.show()
        
        return df_imp