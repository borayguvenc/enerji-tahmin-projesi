
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# src is in current directory
sys.path.append(os.path.join(current_dir, 'src'))

# Import config (local)
# We might need to override some config values dynamically, but we'll import basics
from config import (RAW_TARGET_COL, REPORT_FILENAME)

# Import DataManager and Engine
from src.data_manager import DataManager
from src.reporter import save_detailed_results
from src.hourly_ensemble_manager import HourlyGrandEnsemble

def calculate_mape(y_true, y_pred):
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

def main():
    print("🚀 HOURLY FORWARD TESTING SYSTEM STARTING...\n")
    print("   -> Goal: Forward Test from Sept 2024 to Aug 2025")

    # 1. LOAD FULL DATA
    dm = DataManager()
    # Load all data available, we will slice it ourselves
    full_data = dm.load_and_preprocess()

    # 2. DEFINE TESTING SCHEDULE
    # We want to test starting from 2024-09-01
    # For the first iteration: Train up to 2024-09-01 (exclusive), Test on 2024-09 (inclusive)
    # Then: Train up to 2024-10-01, Test on 2024-10
    # ... until Aug 2025.

    start_date = pd.Timestamp("2024-09-01")
    end_date = pd.Timestamp("2025-09-01") # End of test period (Aug 2025 is last month)
    
    current_test_month = start_date
    
    all_predictions = []
    all_actuals = []
    
    overall_results = pd.DataFrame()

    while current_test_month < end_date:
        # Determine the start of the next month (end of current test month)
        next_month = current_test_month + pd.offsets.MonthBegin(1)
        
        print(f"\n\n====================================================")
        print(f"📅 TESTING PERIOD: {current_test_month.strftime('%Y-%m')}")
        print(f"====================================================")

        # 3. SPLIT TRAIN / TEST
        # Train data: All data BEFORE current_test_month
        # Test data: All data IN current_test_month
        
        train_mask = full_data.index < current_test_month
        test_mask = (full_data.index >= current_test_month) & (full_data.index < next_month)
        
        X_full = full_data.copy()
        y_full = full_data[RAW_TARGET_COL].copy()
        
        # We need to drop target from X if it exists (DataManager might return it in data)
        # DataManager.get_train_test_split logic:
        feature_cols = full_data.select_dtypes(include=['number', 'category', 'bool']).columns.tolist()
        if RAW_TARGET_COL in feature_cols:
            feature_cols.remove(RAW_TARGET_COL)
            
        X = full_data[feature_cols]
        y = full_data[RAW_TARGET_COL]

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_test) == 0:
            print(f"⚠️ No data found for {current_test_month.strftime('%Y-%m')}. Skipping...")
            current_test_month = next_month
            continue

        print(f"   -> Train Size: {len(X_train)} rows")
        print(f"   -> Test Size:  {len(X_test)} rows")

        # 4. INITIALIZE & TRAIN ENGINE
        # Re-initialize engine for each month to simulate fresh retraining
        engine = HourlyGrandEnsemble(project_root=current_dir)
        
        print(f"   -> Training models...")
        engine.fit(X_train, y_train)

        # 5. PREDICT
        print(f"   -> Predicting...")
        y_pred_series = engine.predict(X_test)
        
        # 6. EVALUATE MONTHLY
        # Align indices
        # y_pred_series index should match y_test index
        
        month_mape = calculate_mape(y_test, y_pred_series)
        print(f"   -> 📉 Monthly MAPE: {month_mape:.2f}%")
        
        # Store results
        chunk_df = pd.DataFrame(index=y_test.index)
        chunk_df['Actual'] = y_test
        chunk_df['Prediction'] = y_pred_series
        overall_results = pd.concat([overall_results, chunk_df])
        
        # Move to next month
        current_test_month = next_month

    # 7. OVERALL EVALUATION
    if not overall_results.empty:
        overall_mape = calculate_mape(overall_results['Actual'], overall_results['Prediction'])
        print(f"\n✅ FINAL CUMULATIVE MAPE (Sept 2024 - Aug 2025): {overall_mape:.2f}%")
        
        # 8. PLOTTING
        print("\n--- 📊 GENERATING PLOTS ---")
        plt.figure(figsize=(15, 7))
        plt.plot(overall_results.index, overall_results['Actual'], label='Actual', alpha=0.7, color='blue')
        plt.plot(overall_results.index, overall_results['Prediction'], label='Prediction', alpha=0.7, color='orange')
        plt.title(f'Hourly Energy Prediction - Forward Test (MAPE: {overall_mape:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Energy (MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_filename = "FORWARD_TEST_PLOT.png"
        plt.savefig(os.path.join(current_dir, plot_filename))
        print(f"   -> Plot saved to {plot_filename}")
        
        # Zoomed in plots (one per month?) - Optional, maybe just one big interactive html or just the big static one.
        # Let's save a zoomed plot for the last month as example
        last_month_start = overall_results.index[-1].replace(day=1, hour=0, minute=0, second=0)
        last_month_data = overall_results[overall_results.index >= last_month_start]
        
        plt.figure(figsize=(15, 7))
        plt.plot(last_month_data.index, last_month_data['Actual'], label='Actual', marker='.', linestyle='-')
        plt.plot(last_month_data.index, last_month_data['Prediction'], label='Prediction', marker='x', linestyle='--')
        plt.title(f'Zoomed View: Last Month Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(current_dir, "FORWARD_TEST_ZOOM_LAST_MONTH.png"))

        # 9. REPORT
        print("\n--- 💾 SAVING REPORT ---")
        predictions_pack = {
            'Forward_Test_Hourly': overall_results['Prediction'],
            'Actual': overall_results['Actual']
        }
        
        save_detailed_results(
            y_true=overall_results['Actual'],
            predictions_dict=predictions_pack,
            project_root=current_dir,
            filename="FORWARD_TEST_RESULTS_2024_2025.xlsx"
        )
        print("\nDone.")

    else:
        print("No predictions were made.")

if __name__ == "__main__":
    main()
