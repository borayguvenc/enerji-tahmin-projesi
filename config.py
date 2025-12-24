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
# Normal mod için çıkarılacak sütunlar listesi. 
COLS_TO_DROP = ["Haftanin_gunu_Sin", "Haftanin_gunu_Cos", "Gun_Sin", "Gun_Cos", "Ay_Sin", "Ay_Cos", "Saat_Sin", "Saat_Cos", "Rolling_Mean_3h", "Rolling_Mean_168h", "ÖzelGün_Adı"]


"""


# Özel modlarda deneme amaçlı çıkarılacak sütunlar listesi. 
COLS_TO_DROP = ["Haftanin_gunu_Sin", "Haftanin_gunu_Cos", "Gun_Sin", "Gun_Cos", "Ay_Sin", "Ay_Cos", "Saat_Sin", "Saat_Cos", "Rolling_Mean_3h", 
                "Rolling_Mean_168h", "ADM_Dağıtılan_Enerji_(MWh)_Lag25h",
                "ADM_Dağıtılan_Enerji_(MWh)_Lag26h", "ADM_Dağıtılan_Enerji_(MWh)_Lag27h", "ADM_Dağıtılan_Enerji_(MWh)_Lag504h", 
                "Rolling_Mean_3h_Lag24h","Bulutluluk-MUGLA_MenteseCenter_OpenMeteo_pct", "ÖzelGün_Adı",
                "Bulutluluk-MUGLA_MenteseCenter_OpenMeteo_pct",
                "Bulutluluk-MUGLA_MilasIndustrial_OpenMeteo_pct",
                "Bulutluluk-MUGLA_YataganIndustrial_OpenMeteo_pct",
                "Bulutluluk-MUGLA_SandrasHighAlt_OpenMeteo_pct",
                "Bulutluluk-MUGLA_DalamanPlain_OpenMeteo_pct",
                "Bulutluluk-MUGLA_BodrumCenter_OpenMeteo_pct",
                "Bulutluluk-DNZ_Honaz_OpenMeteo_pct",
                "Bulutluluk-DNZ_OSB_OpenMeteo_pct",
                "Bulutluluk-DNZ_Merkez_OpenMeteo_pct",
                "Bulutluluk-DNZ_IsikliCivril_OpenMeteo_pct",
                "Bulutluluk-AYD_Merkez_OpenMeteo_pct",
                "Bulutluluk-AYD_OSB_OpenMeteo_pct",
                "Bulutluluk-AYD_BuyukMenderes_OpenMeteo_pct",
                "Bulutluluk-AYD_BozdoganMadran_OpenMeteo_pct",
                "MUGLA_MenteseCenter_Dark_Fraction_Pct",
                "MUGLA_MilasIndustrial_Dark_Fraction_Pct",
                "MUGLA_YataganIndustrial_Dark_Fraction_Pct",
                "MUGLA_SandrasHighAlt_Dark_Fraction_Pct",
                "MUGLA_DalamanPlain_Dark_Fraction_Pct",
                "MUGLA_BodrumCenter_Dark_Fraction_Pct",
                "DNZ_Honaz_Dark_Fraction_Pct",
                "DNZ_OSB_Dark_Fraction_Pct",
                "DNZ_Merkez_Dark_Fraction_Pct",
                "DNZ_IsikliCivril_Dark_Fraction_Pct",
                "AYD_Merkez_Dark_Fraction_Pct",
                "AYD_OSB_Dark_Fraction_Pct",
                "AYD_BuyukMenderes_Dark_Fraction_Pct",
                "AYD_BozdoganMadran_Dark_Fraction_Pct"]
        """        



""","ÖzelGün_Adı","DNZ_Honaz_Dark_Fraction_Pct",
    "AYD_BozdoganMadran_Dark_Fraction_Pct",
    "AYD_Merkez_Dark_Fraction_Pct",
    "MUGLA_BodrumCenter_Dark_Fraction_Pct",
    "DNZ_IsikliCivril_Dark_Fraction_Pct",
    "MUGLA_MenteseCenter_Dark_Fraction_Pct",
    "MUGLA_SandrasHighAlt_Dark_Fraction_Pct",
    "DNZ_OSB_Dark_Fraction_Pct",
    "DNZ_Merkez_Dark_Fraction_Pct",
    "MUGLA_MilasIndustrial_Dark_Fraction_Pct",
    "MUGLA_DalamanPlain_Dark_Fraction_Pct",
    "MUGLA_YataganIndustrial_Dark_Fraction_Pct",
    "AYD_OSB_Dark_Fraction_Pct",
    "Bulutluluk-DNZ_Honaz_OpenMeteo_pct",
    "Bulutluluk-AYD_OSB_OpenMeteo_pct",
    "Bulutluluk-DNZ_Merkez_OpenMeteo_pct",
    "ADM_Dağıtılan_Enerji_(MWh)_Lag27h",
    "ADM_Dağıtılan_Enerji_(MWh)_Lag26h",
    "ADM_Dağıtılan_Enerji_(MWh)_Lag25h", "Rolling_Mean_3h_Lag24h"
    """

# --- MODEL PARAMETERS ---
TEST_SIZE = 24 # Last 120 days for testing
WARMUP_PERIOD = 504  # To handle NaN values caused by the largest lag (Lag504)
NUM_OF_SPLITS = 365   # Number of folds for Time Series Cross Validation








