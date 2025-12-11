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
            learning_rate=0.09153183759856734,
            max_depth=5,
            subsample=0.9232356233862338,
            colsample_bytree=0.6011313023146714,
            min_child_weight=5,              # Eklendi
            reg_alpha=3.866820280149056,    # Eklendi (L1 Regularization)
            reg_lambda=2.333871331570596,  # Eklendi (L2 Regularization)
            # ------------------------
            
            objective='reg:squarederror', 
            eval_metric='mae',  
            enable_categorical=True, 
            early_stopping_rounds=50,
            n_jobs=-1
        )

        print(f"[ModelManager] Training started on {len(X_train)} samples...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
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

    def get_feature_importance(self):
        """
        Modelin hangi özelliğe ne kadar önem verdiğini gösterir.
        Hem grafik çizer hem de DataFrame olarak döndürür.
        """
        if self.model is None:
            print("Model henüz eğitilmedi!")
            return None
        
        # Özelliklerin önem skorlarını al (Gain: Bilgi Kazancı)
        # importance_type='gain' -> Hata düşürme gücüne bakar 
        # importance_type='weight' -> Kaç kez kullanıldığına bakar
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        # Sözlükten DataFrame'e çevir ve sırala
        df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain'])
        df_imp = df_imp.sort_values(by='Gain', ascending=False)
        
        plt.figure(figsize=(10, 6))
        # En önemli 20 özelliği çiz
        xgb.plot_importance(self.model, importance_type='gain', max_num_features=20, height=0.5, title='Feature Importance (Gain)')
        plt.show()
        
        return df_imp