import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.model_manager import ModelManager, calculate_mape
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)            
sys.path.append(project_root)



class Evaluator:
    def __init__(self, n_splits=5, test_size=720):
        """
        n_splits: how many folds (assigned in config.py)
        test_size: size of each test set (720 hours = 30 days) (assigned in config.py)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    def run_cross_validation(self, X, y):
        """
        Runs Time Series Cross Validation on given X and y.
        """
        print(f"\n [Evaluator] Cross Validation Başlıyor ({self.n_splits} Fold)...")
        print(f"Her test seti boyutu: {self.test_size} saat\n")
        
        cv_scores = []
        fold = 1

        # TimeSeriesSplit döngüsü
        for train_index, test_index in self.tscv.split(X):
            # 1. Veriyi Böl
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Bilgi bas
            test_start = X_test.index.min()
            test_end = X_test.index.max()
            print(f" Fold {fold}/{self.n_splits} | Test Dönemi: {test_start} -> {test_end}")

            # 2. Her turda SIFIR bir model yöneticisi başlat (Temiz sayfa)
            mm = ModelManager()
            
            # 3. Eğit
            # (eval_set olarak X_test veriyoruz ki early_stopping çalışsın)
            mm.train_model(X_train, y_train, X_test, y_test)
            
            # 4. Tahmin Et
            preds = mm.model.predict(X_test)
            
            # 5. Skor Hesapla
            mape = calculate_mape(y_test, preds)
            cv_scores.append(mape)
            
            print(f"  Fold {fold} MAPE: %{mape:.2f}")
            print("   --------------------------------------------------")
            fold += 1

        return cv_scores

    def print_summary(self, scores):
        """
        Sonuç özetini ekrana basar.
        """
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        print("\n📊 === CROSS VALIDATION SONUÇ RAPORU ===")
        print(f"Tüm Skorlar (MAPE): {['%.2f%%' % s for s in scores]}")
        print(f"Ortalama MAPE     : %{mean_score:.2f}")
        print(f"Standart Sapma    : {std_score:.2f} ")
        print("========================================\n")