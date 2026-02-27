
import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# src is in current directory
sys.path.append(os.path.join(current_dir, 'src'))

# Import config (local)
from config import (NUM_OF_SPLITS, TEST_SIZE, RAW_TARGET_COL, REPORT_FILENAME)

# Import DataManager (copied to local src)
from src.data_manager import DataManager
from src.reporter import save_detailed_results
from src.hourly_ensemble_manager import HourlyGrandEnsemble

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def main():
    print("🚀 HOURLY STRATEGY SYSTEM STARTING...\n")
    print("   -> Goal: Train 24 Separate Grand Ensembles (One per Hour)")
    
    # 1. LOAD DATA
    dm = DataManager()
    # Note: DataManager inside src uses config. So it will use config_hourly if we imported it correctly.
    # However, src.data_manager imports 'config' relative to project structure?
    # No, python imports are tricky. Since we appended 'src' to path, 
    # inside src/data_manager.py: `from config import ...`
    # If we run main_hourly.py from `p0/hourly_strategy/`, `config_hourly.py` is in the same dir.
    # But files inside `src` import `config`. 
    # We must ensure `config.py` is importable as `config`.
    # SOLUTION: We should rename `config_hourly.py` to `config.py` in this folder??
    # OR: We rely on the fact that `sys.path.append(project_root)` in the original files might point to parent.
    # Let's check `src/data_manager.py`:
    #   current_dir = ... src
    #   project_root = ... parent
    #   sys.path.append(project_root)
    #   from config import ...
    # This points to the ORIGINAL config! That's bad because we want to use the NEW input path (../Input).
    #
    # FIX: We should create a dummy `config.py` inside `src` or modify the sys.path behavior.
    # Actually, simpler way: Create `p0/hourly_strategy/config.py` (rename config_hourly to config).
    # And ensure `p0/hourly_strategy` is in sys.path BEFORE `p0`.
    
    dm.load_and_preprocess()
    
    # 2. SPLIT TRAIN/TEST
    # Use the logic from DataManager
    X_train, y_train, X_test, y_test = dm.get_train_test_split()
    
    # 3. INITIALIZE & TRAIN ENGINE
    engine = HourlyGrandEnsemble(project_root=current_dir)
    
    print("\n--- 🧠 TRAINING PHASE START ---")
    engine.fit(X_train, y_train)
    
    # 4. PREDICT
    print("\n--- 🔮 PREDICTION PHASE START ---")
    y_pred = engine.predict(X_test)
    
    # Combine actuals and preds for analysis
    # Ensure indices match
    final_df = pd.DataFrame(index=y_test.index)
    final_df['Actual'] = y_test
    final_df['Prediction'] = y_pred
    
    # 5. EVALUATE
    mape = calculate_mape(final_df['Actual'], final_df['Prediction'])
    print(f"\n✅ FINAL OVERALL MAPE (Combined 24 Hourly Models): {mape:.2f}%")
    
    # 6. REPORT
    print("\n--- 💾 SAVING REPORT ---")
    
    predictions_pack = {
        'Grand_Ensemble_Hourly': final_df['Prediction'],
        'Actual': final_df['Actual'] # Reporter expects predictions, but we can pass whatever
    }
    
    # Calling reporter
    save_detailed_results(
        y_true=final_df['Actual'],
        predictions_dict=predictions_pack,
        project_root=current_dir,
        filename="HOURLY_STRATEGY_RESULTS.xlsx"
    )
    
    print("\nDone.")

if __name__ == "__main__":
    main()
