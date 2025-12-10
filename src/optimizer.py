import optuna
import xgboost as xgb
import numpy as np
import sys
import os
from sklearn.model_selection import TimeSeriesSplit 

# --- YOL AYARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- İMPORTLAR ---
from src.data_manager import DataManager
# Eksik olan config değişkenlerini buraya ekledik:
from config import TEST_SIZE, RAW_TARGET_COL 

# --- YARDIMCI FONKSİYON ---
# Hatayı önlemek için MAPE fonksiyonunu buraya ekledik
def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def objective(trial):
    # 1. Veriyi Yükle ve Hazırla
    dm = DataManager()
    df = dm.load_and_preprocess()
    
    # Hedef sütunu Config'den alıyoruz (Daha güvenli)
    target_col = RAW_TARGET_COL 
    
    # Feature sütunlarını seç (Tarih ve Hedef hariç)
    # Not: DataManager zaten sayısal olmayanları temizlemişti ama garanti olsun
    feature_cols = [c for c in df.columns if c not in [target_col, 'Tarih', 'Datetime']]
    
    X = df[feature_cols]
    y = df[target_col]

    # 2. Optuna'nın Deneyeceği Parametre Aralıkları (Kararlılık Odaklı)
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'hist',
        'enable_categorical': True,
        'n_jobs': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        
        # Değişkenler:
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True)
    }

    # 3. Time Series Cross-Validation ile Kararlılık Testi
    # TEST_SIZE artık import edildiği için hata vermeyecek
    tscv = TimeSeriesSplit(n_splits=5, test_size=TEST_SIZE)
    cv_mape_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        
        # calculate_mape artık tanımlı olduğu için çalışacak
        mape = calculate_mape(y_test, preds)
        cv_mape_scores.append(mape)
        
    # 4. Ortalama MAPE'yi Döndür (Optuna bunu minimize etmeye çalışacak)
    return np.mean(cv_mape_scores) 

if __name__ == "__main__":
    # Kodun çalıştığını görmek için biraz daha detaylı çıktı açalım
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    study = optuna.create_study(direction='minimize')
    print(" Optimizasyon başlıyor (Kararlılık Odaklı - Cross Validated)...")
    
    study.optimize(objective, n_trials=100)
    
    print("\n BEST PARAMETERS:")
    print(study.best_params)
    print(f" BEST CV SCORE (Avg. MAPE): {study.best_value:.4f}")