import json
import os
from datetime import datetime
import numpy as np

class ExperimentLogger:
    def __init__(self, project_root="."):
        # Hata veren yer burasıydı. Artık project_root'u kabul ediyor.
        self.log_dir = os.path.join(project_root, "Reports")
        self.log_path = os.path.join(self.log_dir, "experiment_logs.json")
        
        # Klasör yoksa oluştur
        os.makedirs(self.log_dir, exist_ok=True)

    def log_experiment(self, model_name, mape_scores, config_dict, notes=""):
        # Numpy tiplerini Python tiplerine çevir (JSON hatası almamak için)
        scores_list = [round(float(s), 4) for s in mape_scores] if mape_scores is not None else []
        avg_mape = float(np.mean(scores_list)) if scores_list else 0.0
        min_mape = float(np.min(scores_list)) if scores_list else 0.0
        
        # Config içindeki olası numpy değerlerini de temizle
        clean_config = {}
        for k, v in config_dict.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_config[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                clean_config[k] = v

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "config": clean_config,
            "results": {
                "split_mape_scores": scores_list,
                "average_mape": round(avg_mape, 4),
                "best_split_mape": round(min_mape, 4)
            },
            "notes": notes
        }

        # Mevcut kayıtları oku
        logs = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logs = []

        logs.append(log_entry)

        # Geri kaydet
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        
        print(f"[LOGGER] Deney sonuçları '{self.log_path}' dosyasına eklendi.")