import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import os
import joblib

class CatBoostBayramManager:
    def __init__(self, project_root="."):
        self.model_dir = os.path.join(project_root, "Model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        
        # Derinlik (depth) yüksek, öğrenme hızı (eta) biraz agresif.
        self.params = {
            'iterations': 3000,          # Uzun uzun öğrensin
            'learning_rate': 0.05,       # Sindire sindire
            'depth': 8,                  # Derin baksın (Karmaşık ilişkiler için)
            'l2_leaf_reg': 10,           # Aşırı ezberleyip patlamasın diye fren
            'loss_function': 'MAE',      # Hata farkını direkt düşürmeye çalışsın
            'eval_metric': 'MAPE',
            'random_seed': 42,
            'verbose': 200,
            'allow_writing_files': False
        }

    def _calculate_sniper_weights(self, X_train):
        """
        Normal günler: x1 puan
        Hafta sonu: x5 puan
        Milli Bayramlar: x40 puan
        Ramazan Günleri: x40 puan
        Arife ve Bayramlar: x100 puan!
        """
        weights = np.ones(len(X_train))
        

        """
        # 1. Hafta Sonu (Biraz önemli)
        if 'Haftanın_Günü' in X_train.columns:
            # 5: Cumartesi, 6: Pazar (Eğer 0-6 ise 5-6, 1-7 ise 6-7 kontrol et)
            weekend_mask = X_train['Haftanın_Günü'].isin([5, 6])
            weights[weekend_mask] = 5.0
        """

        # 2. Milli Bayramlar (Önemli)
        if 'Milli_Bayram' in X_train.columns:
            weights[X_train['Milli_Bayram'] == 1] = 40.0

        if 'Is_Ramadan' in X_train.columns:
            weights[X_train['Is_Ramadan'] == 1] = 10.0
            
        # 3. Dini Bayramlar
        # Ramazan ve Kurban Bayramı'na devasa ağırlık veriyoruz.
        if 'Ramazan_Bayram' in X_train.columns:
            weights[X_train['Ramazan_Bayram'] == 1] = 100.0
            
        if 'Kurban_Bayram' in X_train.columns:
            weights[X_train['Kurban_Bayram'] == 1] = 1.0
            
        # Varsa Arife günleri (Genelde bayramdan önceki gün düşüş başlar)

        print(f"[Sniper] Ağırlıklar ayarlandı. Max Ağırlık: {weights.max()}")
        return weights

    def train_model(self, X_train, y_train, X_test, y_test):
        
        # 1. Kategorik Değişkenleri Bul (CatBoost sever)
        cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']
        if len(cat_features) > 0:
            print(f"   -> Kategorik sütunlar işleniyor: {cat_features}")
        
        # 2. Ağırlıkları Hesapla (Büyü burada!)
        sniper_weights = self._calculate_sniper_weights(X_train)
        
        # 3. Veri Havuzunu Oluştur (Pool)
        train_pool = Pool(X_train, y_train, cat_features=cat_features, weight=sniper_weights)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)
        
        # 4. Modeli Başlat ve Eğit
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=100, # 100 adım iyileşmezse dur
            use_best_model=True
        )
        print("✅ [Sniper] Eğitim Tamamlandı.")

    def predict(self, X):
        if self.model is None:
            raise Exception("Model eğitilmemiş kanks! Önce train_model çalıştır.")
        return self.model.predict(X)

    def save_model(self, filename="catboost_sniper.cbm"):
        path = os.path.join(self.model_dir, filename)
        self.model.save_model(path)
        print(f"💾 [Sniper] Model silahı şuraya kilitlendi: {path}")

    def load_model(self, filename="catboost_sniper.cbm"):
        path = os.path.join(self.model_dir, filename)
        self.model = CatBoostRegressor()
        self.model.load_model(path)
        print(f"📂 [Sniper] Model göreve hazır: {path}")