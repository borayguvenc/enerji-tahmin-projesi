# Energy Output Forecasting Project

## Architecture Overview

This project is built with a modular, object-oriented pipeline designed for predicting distributed energy (MWh). It features advanced feature engineering, automated hyperparameter optimization, and ensemble machine learning models (XGBoost, LightGBM, CatBoost, ANN, and a Meta-Stacking Regressor).

### Core Components

*   **`config.py`**: The control center of the project. Contains file paths, target column definitions (`RAW_TARGET_COL`), start/end dates for data filtering, model hyperparameters, and lists of columns to drop during preprocessing (`COLS_TO_DROP`).
*   **`src/data_manager.py`**: Responsible for loading the raw Excel data, preprocessing dates, generating time-based features, handling holiday and lockdown flags, and engineering thermal/weather features (such as Heating/Cooling Degree Days - HDD/CDD). It also cleanly splits the data into train and test sets.
*   **`src/optimizer.py`**: Handles hyperparameter tuning using the `optuna` library. It can optimize XGBoost, LightGBM, and CatBoost models. It supports both general rolling optimization and target-date specific optimization.
*   **`src/evaluator.py`**: Conducts robust Time Series Cross-Validation to evaluate model performance without data leakage. Computes Mean Absolute Percentage Error (MAPE) across folds and manages different evaluation configurations.
*   **`src/model_manager.py` (and specific managers)**: Object-oriented wrappers for each machine learning algorithm (e.g., `lightgbm_manager.py`, `catboost_manager.py`, `ann_manager.py`). They handle model initialization, training, evaluation metrics generation, and saving/loading the model artifacts.
*   **`src/stacking_manager.py`**: Implements a meta-learner (Ridge Regression) used in the 'ALL' mode to smartly combine the predictions of XGBoost, LightGBM, and CatBoost into a more resilient "Grand Ensemble" hybrid prediction.
*   **`main.py`**: The primary coordination script. Based on the selected operating mode, it orchestrates data loading, cross-validation, final model training, experiment logging, and generates detailed evaluation exports.

---

## How to Use the Project

### 1. Configuration & Data Setup
Before running the pipelines, ensure your data and parameters are set correctly in `config.py`.
*   Update the `INPUT_FILE_PATH` to point to your raw data (typically `Input/INPUT_AYDEM_UPDATED.xlsx`).
*   Adjust `RAW_TARGET_COL` if the target variable name differs.
*   Modify `COLS_TO_DROP` to exclude any columns not needed for the current iteration.

### 2. Hyperparameter Optimization (`optimizer.py`)
To find the best hyperparameters for a specific model before beginning the main training phase, use the optimizer tool.

Optimize a specific model globally:
```bash
python src/optimizer.py --model XGB --trials 50
```

Optimize for a specific target time-frame (e.g., a specific month):
```bash
python src/optimizer.py --model LGBM --trials 100 --start_date 2025-01-01 --end_date 2025-01-31
```
*Optimized parameter settings are automatically saved locally into JSON files (e.g., `best_params_lgbm_general.json`).*

### 3. Running the Full Pipeline (`main.py`)
Run the primary script to trigger data preprocessing, cross-validation, and final training.

```bash
python main.py --mode ALL
```

**Available Operating Modes:**
*   `XGB`: Runs only the XGBoost pipeline.
*   `LGBM`: Runs only the LightGBM pipeline.
*   `CAT`: Runs only the CatBoost pipeline.
*   `CAT_TUNED`: Runs the heuristically tuned version of CatBoost.
*   `ALL`: Runs the **Grand Ensemble** mode. Trains XGBoost, LightGBM, and CatBoost, then routes their outputs through the `StackingManager` to find the optimal combined prediction.
*   `ANN`: Runs the Artificial Neural Network pipeline.
*   `SNIPER`: Evaluates a specialized model configuration tuned specifically for holiday and special event periods.

*Note: If you run `python main.py` without the `--mode` argument attached, the application will provide an interactive prompt for you to select the operation mode.*

### 4. Viewing Results and Outputs
*   **Excel Reports**: Detailed predictions and aggregated forecasting results are generated during execution and saved as Excel files (defined by `REPORT_FILENAME` in `config.py`).
*   **Experiment Logs**: Run history, model specifications, CV scores, and MAPE metrics are automatically logged by `src/experiment_logger.py` for comparative tracking.
*   **Saved Models**: The trained models are dumped structurally into the `Model/` directory for later inference deployments.
