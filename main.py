import sys
import os

# src klasörünü Python yoluna ekle (böylece importlar çalışır)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

from src.data_manager import DataManager
from src.model_manager import ModelManager

def main():
    print("SİSTEM BAŞLATILIYOR...\n")

    # Veri hazırlığı
    dm = DataManager()
    
    # Veriyi yükle ve temizle
    dm.load_and_preprocess()
    
    # Train/Test ayrımını yap (Varsayılan: Son 1 ay test)
    X_train, y_train, X_test, y_test = dm.get_train_test_split()

    # Model Eğitimi
    mm = ModelManager()
    
    # Modeli eğit
    mm.train_model(X_train, y_train, X_test, y_test)

    # Model Değerlendirme
    mm.evaluate(X_test, y_test)
    
    # Modeli kaydet
    mm.save_model()
    
    print("\n İŞLEM TAMAMLANDI.")

if __name__ == "__main__":
    main()