
import os

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to find the Input folder (since we are in p0/hourly_strategy)
INPUT_FILE_PATH = os.path.join(BASE_DIR, '..', 'Input', 'INPUT_AYDEM_UPDATED.xlsx')

# --- MODEL NAME ---
MODEL_NAME = 'xgboost_hourly.json'

# --- CRITICAL COLUMN NAMES ---
RAW_TARGET_COL = "ADM_Dağıtılan_Enerji_(MWh)"
RAW_DATE_COL = "Tarih"
RAW_HOUR_COL = "Saat"

DATA_START_DATE = None 
DATA_END_DATE   = None 

REPORT_FILENAME = "YILLIK_DENEME_Regressyon_Hourly.xlsx"

# --- DROP LIST ---
# Updated to drop columns that are full of NaNs
COLS_TO_DROP = [
    # Fixed Missing Comma
    "After_Bayram", "Haftanin_gunu_Sin", 
    "Haftanin_gunu_Cos", "Gun_Sin", "Gun_Cos", "Ay_Sin", "Ay_Cos", "Saat_Sin", "Saat_Cos", 
    "Rolling_Mean_3h", "Rolling_Mean_168h", "ÖzelGün_Adı",
    
     # Weather actual
    "AYDIN_BozdoganMadran_app_temp_actual", "AYDIN_BuyukMenderes_app_temp_actual", "AYDIN_OSB_app_temp_actual",
    "DENIZLI_Honaz_app_temp_actual", "DENIZLI_IsikliCivril_app_temp_actual","DENIZLI_OSB_app_temp_actual", "MUGLA_DalamanPlain_app_temp_actual",
    "MUGLA_SandrasHighAlt_app_temp_actual", "MUGLA_YataganIndustrial_app_temp_actual",

    # --- Cloud Cover Actuals ---
    "AYDIN_BozdoganMadran_cloud_actual", "AYDIN_BuyukMenderes_cloud_actual",
    "AYDIN_OSB_cloud_actual",
    "DENIZLI_Honaz_cloud_actual", "DENIZLI_IsikliCivril_cloud_actual",
    "DENIZLI_OSB_cloud_actual",
    "MUGLA_DalamanPlain_cloud_actual",
    "MUGLA_SandrasHighAlt_cloud_actual", "MUGLA_YataganIndustrial_cloud_actual",

    # --- Precipitation Actuals ---
    "AYDIN_BozdoganMadran_precip_actual", "AYDIN_BuyukMenderes_precip_actual",
    "AYDIN_OSB_precip_actual",
    "DENIZLI_Honaz_precip_actual", "DENIZLI_IsikliCivril_precip_actual",
    "DENIZLI_OSB_precip_actual",
    "MUGLA_DalamanPlain_precip_actual",
    "MUGLA_SandrasHighAlt_precip_actual", "MUGLA_YataganIndustrial_precip_actual",
    
    # Weather forecasts
    "AYDIN_BozdoganMadran_app_temp_fc", "AYDIN_BuyukMenderes_app_temp_fc",
    "AYDIN_Merkez_app_temp_fc", "AYDIN_OSB_app_temp_fc",
    "DENIZLI_Honaz_app_temp_fc", "DENIZLI_IsikliCivril_app_temp_fc",
    "DENIZLI_Merkez_app_temp_fc", "DENIZLI_OSB_app_temp_fc",
    "MUGLA_BodrumCenter_app_temp_fc", "MUGLA_DalamanPlain_app_temp_fc",
    "MUGLA_MenteseCenter_app_temp_fc", "MUGLA_MilasIndustrial_app_temp_fc",
    "MUGLA_SandrasHighAlt_app_temp_fc", "MUGLA_YataganIndustrial_app_temp_fc",

    # --- Cloud Cover Forecasts ---
    "AYDIN_BozdoganMadran_cloud_fc", "AYDIN_BuyukMenderes_cloud_fc",
    "AYDIN_Merkez_cloud_fc", "AYDIN_OSB_cloud_fc",
    "DENIZLI_Honaz_cloud_fc", "DENIZLI_IsikliCivril_cloud_fc",
    "DENIZLI_Merkez_cloud_fc", "DENIZLI_OSB_cloud_fc",
    "MUGLA_BodrumCenter_cloud_fc", "MUGLA_DalamanPlain_cloud_fc",
    "MUGLA_MenteseCenter_cloud_fc", "MUGLA_MilasIndustrial_cloud_fc",
    "MUGLA_SandrasHighAlt_cloud_fc", "MUGLA_YataganIndustrial_cloud_fc",

    # --- Precipitation Forecasts ---
    "AYDIN_BozdoganMadran_precip_fc", "AYDIN_BuyukMenderes_precip_fc",
    "AYDIN_Merkez_precip_fc", "AYDIN_OSB_precip_fc",
    "DENIZLI_Honaz_precip_fc", "DENIZLI_IsikliCivril_precip_fc",
    "DENIZLI_Merkez_precip_fc", "DENIZLI_OSB_precip_fc",
    "MUGLA_BodrumCenter_precip_fc", "MUGLA_DalamanPlain_precip_fc",
    "MUGLA_MenteseCenter_precip_fc", "MUGLA_MilasIndustrial_precip_fc",
    "MUGLA_SandrasHighAlt_precip_fc", "MUGLA_YataganIndustrial_precip_fc" 
]

# --- MODEL PARAMETERS ---
TEST_SIZE = 24
WARMUP_PERIOD = 504  
NUM_OF_SPLITS = 365
