import os

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(BASE_DIR, 'Input', 'INPUT_AYDEM.xlsx') 

# --- CRITICAL COLUMN NAMES (Must match Excel headers) ---
RAW_TARGET_COL = "ADM_Dağıtılan_Enerji_(MWh)"
RAW_DATE_COL = "Tarih"
RAW_HOUR_COL = "Saat"

# --- DROP LIST ---
COLS_TO_DROP = ["Ay_Cos","Saat_Cos", "Ay_Sin", "Saat_Sin"]

# --- MODEL PARAMETERS ---
TEST_SIZE = 24 * 365  # Last 120 days for testing
WARMUP_PERIOD = 504  # To handle NaN values caused by the largest lag (Lag504)