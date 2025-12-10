import os

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(BASE_DIR, 'Input', 'INPUT_AYDEM.xlsx') 

# --- MODEL NAME ---
MODEL_NAME = 'xgboost_model.json'    

# --- CRITICAL COLUMN NAMES (Must match Excel headers) ---
RAW_TARGET_COL = "ADM_Dağıtılan_Enerji_(MWh)"
RAW_DATE_COL = "Tarih"
RAW_HOUR_COL = "Saat"

# --- DROP LIST ---
COLS_TO_DROP = ["Haftanin_gunu_Sin", "Haftanin_gunu_Cos", "Gun_Sin", "Gun_Cos", "Ay_Sin", "Ay_Cos", "Saat_Sin", "Saat_Cos", "Rolling_Mean_3h", "Rolling_Mean_168h"]

# --- MODEL PARAMETERS ---
TEST_SIZE = 24 * 30  # Last 120 days for testing
WARMUP_PERIOD = 504  # To handle NaN values caused by the largest lag (Lag504)
NUM_OF_SPLITS = 12  # Number of folds for Time Series Cross Validation