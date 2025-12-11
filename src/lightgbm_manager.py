import lightgbm as lgb
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

from config import MODEL_NAME

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

class LightGBMManager:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.join(project_root, 'Model')
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train, y_train, X_test, y_test):
        print("[LightGBMManager] Initializing LightGBM Regressor...")
        
        # LightGBM özel veri formatını sever ama pandas ile de çalışır.
        # Kategorik değişkenleri otomatik tanır ama biz int'e çevirdiğimiz için sorun yok.
        
        self.model = lgb.LGBMRegressor(
            # 1. Sabit Ayarlar
            n_estimators=2000,       # Kapasiteyi yüksek tutuyoruz, early_stopping nerede duracağını bilir.
            objective='regression',
            n_jobs=-1,
            random_state=42,
            importance_type='gain',
            
            learning_rate=0.05074154948325871,
            num_leaves=30,           # Ağaç karmaşıklığı
            max_depth=6,             # Derinlik limiti (Overfitting freni)
            min_child_samples=28,    # Bir yaprakta en az 28 veri olsun
            subsample=0.813694299293671,         # Satırların %81'ini kullan
            colsample_bytree=0.6293259247827979, # Sütunların %63'ünü kullan (Çeşitlilik için süper)
            reg_alpha=7.73076167663075,          # L1 Regularization (Gürültü temizliği)
            reg_lambda=2.116570077959441         # L2 Regularization
        )

        print(f"[LightGBMManager] Training started on {len(X_train)} samples...")
        
        # LightGBM eğitim formatı
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mae',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        print("[LightGBMManager] Training finished.")

    def evaluate(self, X_test, y_test):
        if self.model is None:
            print("Model is not trained yet!")
            return

        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = calculate_mape(y_test, preds)
        
        print("\n--- LightGBM Performance ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : %{mape:.2f}") 
        print("----------------------------\n")
        
        return preds

    def get_feature_importance(self):
        if self.model is None:
            return None
        
        # LightGBM Feature Importance çizimi
        lgb.plot_importance(self.model, max_num_features=20, importance_type='gain', figsize=(10, 6), title='LightGBM Feature Importance (Gain)')
        plt.show()


    def save_model(self, filename='model_lgbm.txt'):
        """
        LightGBM modelini güvenli bir şekilde (Türkçe karakter sorunu olmadan) kaydeder.
        """
        if self.model is None:
            print("Model eğitilmedi, kayıt yapılamıyor.")
            return

        save_path = os.path.join(self.model_dir, filename)
        
        try:
            # YÖNTEM 1 (GÜVENLİ): Modeli string olarak al, Python ile biz kaydedelim.
            # Bu sayede "Masaüstü" gibi Türkçe yollarda hata vermez.
            model_str = self.model.booster_.model_to_string()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(model_str)
                
            print(f"[LightGBMManager] Model başarıyla kaydedildi: {save_path}")
            
        except Exception as e:
            # Eğer yukarıdaki çalışmazsa eski yöntemi dener ama hata basar
            print(f"[LightGBMManager] Kayıt sırasında hata oluştu: {e}")