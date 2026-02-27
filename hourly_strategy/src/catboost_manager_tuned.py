import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- YOL AYARLARI ---
# Proje ana dizinini bul ve sys.path'e ekle (Config dosyasına erişim için)
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

class CatBoostManagerTuned:
    def __init__(self):
        self.model = None
        self.model_dir = os.path.join(project_root, 'Model')
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        CatBoost modelini eğitir.
        """
        print("[CatBoostManager] Initializing Tuned CatBoost Regressor...")
        
        # 1. OTOMATİK KATEGORİK SÜTUN TESPİTİ
        # Veri setindeki 'category' veya 'object' tipindeki sütunları bulur
        cat_features_indices = np.where(X_train.dtypes != float)[0] # Float olmayan her şeyi kategorik sayabiliriz
        # Veya daha garantisi isim olarak bulmak:
        cat_features_names = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_features_names:
            print(f"[CatBoostManager] Kategorik sütunlar tespit edildi: {cat_features_names}")
        
        # CatBoost Parametreleri
        self.model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.05,
            depth=6,
            loss_function='MAE',
            eval_metric='MAPE',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            allow_writing_files=False,
            cat_features=cat_features_names 
        )

        print(f"[CatBoostManager] Training started on {len(X_train)} samples...")

        
        # --- ADIM 2: CEZALARI ARTIR (Katsayılar) ---
        weights = np.ones(len(X_train))
        # A. Ramazan Günleri (Örn: 3 Kat Önemli)
        if 'Is_Ramadan' in X_train.columns:
            # Sütunu bul ve maske oluştur
            mask = (X_train['Is_Ramadan'] == 1).values
            weights[mask] *= 5.0
            
        # B. Sahur Saatleri (Örn: 5 Kat Önemli - Gece hatalarını düzeltmek için)
        if 'Is_Sahur' in X_train.columns:
            mask = (X_train['Is_Sahur'] == 1).values
            weights[mask] *= 2.0
            
        # C. Milli/Dini Bayramlar (Örn: 10 Kat Önemli - Mart sonundaki çukuru düzeltmek için)
        if 'Milli_Bayram' in X_train.columns: # Sütun adın 'Is_Holiday' ise onu yaz
            mask = (X_train['Milli_Bayram'] == 1).values
            weights[mask] *= 3.0
            
        if 'Kurban_Bayram' in X_train.columns:
            mask = (X_train['Kurban_Bayram'] == 1).values
            weights[mask] *= 5.0
        
        if 'Ramazan_Bayram' in X_train.columns:
            mask = (X_train['Ramazan_Bayram'] == 1).values
            weights[mask] *= 5.0
        
        
        if 'Is_Eve' in X_train.columns: 
            mask = (X_train['Is_Eve'] == 1).values
            weights[mask] *= 5.0

        if 'After_Bayram' in X_train.columns:
            mask = (X_train['After_Bayram'] == 1).values
            weights[mask] *= 5.0

        if 'Yilbasi' in X_train.columns:
            mask = (X_train['Yilbasi'] == 1).values
            weights[mask] *= 5.0

       
        
        # Veriyi hazırla (Eğitim ve Test için ayrı ayrı)
        train_dataset = Pool(data=X_train, label=y_train, weight=weights, cat_features=cat_features_names)
        test_dataset = Pool(data=X_test, label=y_test, cat_features=cat_features_names)

        # Eğitimi başlat
        self.model.fit(train_dataset, eval_set=test_dataset, use_best_model=True)
        print("[CatBoostManager] Training finished.")



    def evaluate(self, X_test, y_test):
        """
        Modeli test seti üzerinde değerlendirir ve metrikleri basar.
        """
        if self.model is None:
            print("Model is not trained yet!")
            return None

        print("[CatBoostManager] Predicting...")
        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = calculate_mape(y_test, preds)
        
        print("\n--- CatBoost Performance ---")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAPE : %{mape:.2f}") 
        print("----------------------------\n")
        
        return preds

    def get_feature_importance(self):
        """
        Modelin hangi özelliğe ne kadar önem verdiğini grafik olarak çizer.
        """
        if self.model is None:
            print("Model is not trained yet!")
            return
        
        feature_importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        # Önem sırasına göre diz (Küçükten büyüğe)
        sorted_idx = np.argsort(feature_importance)
        
        # En önemli son 20 özelliği göster
        top_n = 20
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx[-top_n:])), feature_importance[sorted_idx][-top_n:], align='center')
        plt.yticks(range(len(sorted_idx[-top_n:])), [feature_names[i] for i in sorted_idx][-top_n:])
        plt.xlabel('Feature Importance')
        plt.title('CatBoost Feature Importance')
        plt.tight_layout()
        plt.show()

    def save_model(self, filename='best_catboost_model.cbm'):
        """
        Eğitilen modeli kaydeder.
        """
        if self.model is None:
            print("Model eğitilmedi, kayıt yapılamıyor.")
            return

        save_path = os.path.join(self.model_dir, filename)
        self.model.save_model(save_path)
        print(f"[CatBoostManager] Model saved to: {save_path}")