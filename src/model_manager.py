import xgboost as xgb
import os
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIG YOLU AYARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

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
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
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

    def save_model(self, filename='xgboost_model_v3_365.json'):
        if self.model is None:
            print("No model to save.")
            return

        save_path = os.path.join(self.model_dir, filename)
        self.model.save_model(save_path)
        print(f"[ModelManager] Model saved to: {save_path}")

    def load_model(self, filename='xgboost_model_v3_365.json'):
        load_path = os.path.join(self.model_dir, filename)
        if os.path.exists(load_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(load_path)
            print(f"[ModelManager] Model loaded from: {load_path}")
        else:
            print(f"Model file not found at: {load_path}")