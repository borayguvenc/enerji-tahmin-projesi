import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV

class StackingManager:
    def __init__(self, project_root="."):
        self.model_dir = os.path.join(project_root, 'Model')
        os.makedirs(self.model_dir, exist_ok=True)
        # RidgeCV: En iyi katsayıları otomatik seçen regresyon modeli
        self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])

    def train_meta_model(self, full_df):
        """
        Base modellerin CV tahminlerini girdi (X), 
        gerçek değerleri hedef (y) olarak kullanarak eğitir.
        """
        print("\n[StackingManager] 2. Katman Regresyonu Eğitiliyor...")
        
        # Girdiler: Diğer modellerin tahminleri
        X_meta = full_df[['XGB_Pred', 'LGBM_Pred', 'CAT_Pred']]
        y_meta = full_df['Actual']
        
        self.meta_model.fit(X_meta, y_meta)
        
        # Katsayıları yazdır (Hangi modele ne kadar güvenildiğini gösterir)
        coefs = dict(zip(X_meta.columns, np.round(self.meta_model.coef_, 4)))
        print(f"   -> Model Katsayıları (Ağırlıklar): {coefs}")
        print(f"   -> Bias (Sabit Kayma): {self.meta_model.intercept_:.4f}")

    def predict_hybrid(self, full_df):
        """Eğitilen katsayılarla nihai hibrit tahmini üretir."""
        X_meta = full_df[['XGB_Pred', 'LGBM_Pred', 'CAT_Pred']]
        return self.meta_model.predict(X_meta)

    def save_model(self, filename="stacking_ridge.joblib"):
        path = os.path.join(self.model_dir, filename)
        joblib.dump(self.meta_model, path)
        print(f"[StackingManager] Meta-model kaydedildi: {path}")