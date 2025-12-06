import sys
import os

# src klasörünü yola ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from config import (NUM_OF_SPLITS, TEST_SIZE)

from src.data_manager import DataManager
from src.model_manager import ModelManager
from src.evaluator import Evaluator
from config import RAW_TARGET_COL

def main():
    print(" SİSTEM BAŞLATILIYOR...\n")

    # 1. VERİ YÜKLEME
    dm = DataManager()
    dm.load_and_preprocess()
    
    # Tüm veriyi hazırla (Cross Validation için tamamı lazım)
    df = dm.data
    
    # Feature ve Target ayrımı
    feature_cols = df.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
    if RAW_TARGET_COL in feature_cols:
        feature_cols.remove(RAW_TARGET_COL)
        
    X = df[feature_cols]
    y = df[RAW_TARGET_COL]

    # 2. CROSS VALIDATION (EVALUATOR KULLANIMI)
    evaluator = Evaluator(n_splits=NUM_OF_SPLITS, test_size=TEST_SIZE) # İstersen n_splits=12 yapıp 1 yılı test et
    
    # Testi çalıştır
    scores = evaluator.run_cross_validation(X, y)
    
    # Sonuçları raporla
    evaluator.print_summary(scores)
    
    # 1. Son 30 günü test, gerisini train yapalım (Tek seferlik)
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    mm_final = ModelManager()
    mm_final.train_model(X_train, y_train, X_test, y_test)
    
    # 2. ÖZELLİK ÖNEMİNİ GÖSTER
    mm_final.get_feature_importance() # <-- İŞTE BU SATIR SANA GRAFİĞİ VERECEK

if __name__ == "__main__":
    main()