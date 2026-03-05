"""
STACKING MANAGER — Sızıntısız (Leakage-Free) Ensemble Yöneticisi
================================================================

SORUN NE İDİ?
─────────────
Eski kod, 3 modelin (XGB, LGBM, CAT) CV tahminlerini topluyor, sonra bir 
Ridge meta-model'i BU TAHMİNLER ÜZERİNDE eğitiyordu. Ardından AYNI VERİ 
üzerinde tahmin yapıp "Hibrit MAPE" olarak raporluyordu.

Bu, tren=test sızıntısıdır (data leakage). Model kendi eğitim verisini 
tahmin ettiği için MAPE yapay olarak düşük çıkıyordu.

ÇÖZÜM NE?
─────────
3 farklı sızıntısız (leakage-free) strateji:

1. EXPANDING-WINDOW STACKING (Genişleyen Pencere)
   ├── İlk 10 fold: Sadece veri topla (meta-model için yeterli veri yok)
   ├── Fold 11: Fold 1-10'daki tahminlerle Ridge eğit → Fold 11'i tahmin et
   ├── Fold 12: Fold 1-11'deki tahminlerle Ridge eğit → Fold 12'yi tahmin et
   └── ...böyle devam eder. Her fold'un tahmini GERÇEKTEN out-of-sample'dır.

2. INVERSE-MAPE WEIGHTING (Ters MAPE Ağırlıklandırma)
   ├── Her fold için, önceki fold'lardaki model bazlı MAPE'leri hesapla
   ├── Hangi model daha düşük MAPE → ona daha fazla ağırlık ver
   ├── Formül: ağırlık = (1/MAPE_model) / toplam(1/MAPE_hepsi)
   └── Örnek: XGB %3, LGBM %2, CAT %2.5 ise → LGBM en çok ağırlık alır

3. CONSTRAINED OPTIMIZATION (Kısıtlı Optimizasyon)
   ├── scipy.optimize ile optimal ağırlıkları bul
   ├── Kısıtlar: tüm ağırlıklar ≥ 0 ve toplamları = 1
   └── Hedef fonksiyon: önceki fold'lardaki MAPE'yi minimize et

Sistem 3 stratejiyi + basit ortalamayı karşılaştırır ve EN İYİSİNİ seçer.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy.optimize import minimize


def calculate_mape(y_true, y_pred):
    """MAPE (Mean Absolute Percentage Error) hesaplar."""
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


class StackingManager:
    """
    Sızıntısız Ensemble Yöneticisi.
    
    3 modelin (XGB, LGBM, CAT) CV tahminlerini alır ve 
    en iyi birleştirme stratejisini otomatik seçer.
    """

    # Meta-model eğitimi başlamadan önce kaç fold biriksin?
    # Çok az fold ile Ridge/optimizasyon güvenilmez olur.
    MIN_WARMUP_FOLDS = 10

    # 3 base modelin tahmin sütun isimleri
    PRED_COLS = ['XGB_Pred', 'LGBM_Pred', 'CAT_Pred']

    def __init__(self, project_root="."):
        self.model_dir = os.path.join(project_root, 'Model')
        os.makedirs(self.model_dir, exist_ok=True)
        self.meta_model = None      # Final Ridge model (üretime verilecek)
        self.best_method = None     # Kazanan stratejinin adı
        self.best_weights = None    # Kazanan stratejinin ağırlıkları

    # ==================================================================
    # ANA FONKSİYON — main.py'den çağrılır
    # ==================================================================
    def run_ensemble(self, full_df):
        """
        CV sonuçlarını (full_df) alır, 4 stratejiyi çalıştırır,
        en düşük MAPE'li olanı seçer.
        
        Parametreler:
            full_df: Evaluator'dan gelen DataFrame
                     Sütunlar: Actual, XGB_Pred, LGBM_Pred, CAT_Pred, fold_id
        
        Döndürür:
            best_predictions: En iyi stratejinin tahminleri (Series)
            results_dict: {strateji_adı: (mape, tahminler)} sözlüğü
        """
        print("\n" + "=" * 60)
        print("  SIZINTISIZ ENSEMBLE KARŞILAŞTIRMASI")
        print("=" * 60)

        results = {}

        # ── Strateji 1: Basit Ortalama (mevcut yöntem, sızıntı riski yok) ──
        # 3 modelin tahminlerinin aritmetik ortalaması.
        # En basit yöntem ama modeller arası performans farkını yok sayar.
        simple_preds = full_df[self.PRED_COLS].mean(axis=1)
        simple_mape = calculate_mape(full_df['Actual'], simple_preds)
        results['Simple_Average'] = (simple_mape, simple_preds)
        print(f"\n  [1] Basit Ortalama MAPE:              %{simple_mape:.4f}")

        # ── Strateji 2: Genişleyen Pencere Stacking (Ridge) ──
        # Her fold'da, önceki fold'ların tahminleriyle Ridge eğiterek
        # mevcut fold'u tahmin eder. Gerçek out-of-sample.
        stacking_preds, stacking_mape = self._expanding_window_stacking(full_df)
        if stacking_preds is not None:
            results['Stacking_Ridge'] = (stacking_mape, stacking_preds)
            print(f"  [2] Expanding Stacking MAPE:          %{stacking_mape:.4f}")
        else:
            print(f"  [2] Expanding Stacking:               ATLANDI (yeterli fold yok)")

        # ── Strateji 3: Ters MAPE Ağırlıklandırma ──
        # Düşük MAPE'li modele daha fazla söz hakkı verir.
        # Meta-model yok, sadece basit ağırlıklı ortalama.
        inv_preds, inv_mape = self._inverse_mape_weighting(full_df)
        if inv_preds is not None:
            results['Inverse_MAPE'] = (inv_mape, inv_preds)
            print(f"  [3] Ters MAPE Ağırlıklandırma MAPE:   %{inv_mape:.4f}")
        else:
            print(f"  [3] Ters MAPE Ağırlıklandırma:        ATLANDI (yeterli fold yok)")

        # ── Strateji 4: Kısıtlı Optimizasyon (scipy) ──
        # Matematiksel optimizasyonla en iyi ağırlıkları bulur.
        # Kısıtlar: ağırlıklar ≥ 0 ve toplamı = 1
        opt_preds, opt_mape = self._constrained_optimization(full_df)
        if opt_preds is not None:
            results['Optimized_Weights'] = (opt_mape, opt_preds)
            print(f"  [4] Kısıtlı Optimizasyon MAPE:        %{opt_mape:.4f}")
        else:
            print(f"  [4] Kısıtlı Optimizasyon:             ATLANDI (yeterli fold yok)")

        # ── Kazananı Seç (en düşük MAPE) ──
        best_name = min(results, key=lambda k: results[k][0])
        best_mape, best_preds = results[best_name]
        self.best_method = best_name

        print(f"\n  🏆 KAZANAN: {best_name}  (MAPE = %{best_mape:.4f})")
        print("=" * 60 + "\n")

        # Üretim (production) için final Ridge modelini TÜM OOF verisi
        # üzerinde eğit. Bu sızıntı DEĞİL çünkü OOS performansını 
        # yukarıda zaten kanıtladık.
        self._train_final_meta_model(full_df)

        return best_preds, results

    # ==================================================================
    # STRATEJİ 1: GENİŞLEYEN PENCERE STACKING
    # ==================================================================
    def _expanding_window_stacking(self, full_df):
        """
        Mantık:
        ───────
        Diyelim 12 fold var. İlk 10 fold'u "ısınma" olarak kullan.
        
        Fold 11 için: 
          → Ridge'i fold 1-10'un tahminleriyle eğit
          → Fold 11'i tahmin et (model bu veriyi hiç GÖRMEDİ)
        
        Fold 12 için:
          → Ridge'i fold 1-11'in tahminleriyle eğit
          → Fold 12'yi tahmin et
        
        Böylece her tahmin gerçekten out-of-sample olur.
        Ridge regresyon (L2 cezalı) aşırı öğrenmeyi engeller.
        """
        if 'fold_id' not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df['fold_id'].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        # Sonuçları burada biriktiriyoruz
        stacking_parts = []

        for i, fid in enumerate(fold_ids):
            # İlk MIN_WARMUP_FOLDS fold'u atla (yeterli eğitim verisi yok)
            if i < self.MIN_WARMUP_FOLDS:
                continue

            # Eğitim verisi = mevcut fold'dan ÖNCEKİ tüm fold'lar
            train_mask = full_df['fold_id'].isin(fold_ids[:i])
            test_mask = full_df['fold_id'] == fid

            # X: 3 modelin tahminleri, Y: gerçek değerler
            X_meta_train = full_df.loc[train_mask, self.PRED_COLS]
            y_meta_train = full_df.loc[train_mask, 'Actual']
            X_meta_test = full_df.loc[test_mask, self.PRED_COLS]

            # RidgeCV: L2 regularizasyon parametresini (alpha) otomatik seçer
            ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_meta_train, y_meta_train)
            preds = ridge.predict(X_meta_test)

            fold_preds = pd.Series(preds, index=X_meta_test.index)
            stacking_parts.append(fold_preds)

        # Tüm fold tahminlerini birleştir
        all_stacking_preds = pd.concat(stacking_parts)

        # Sadece stacking tahminleri olan fold'lar üzerinde MAPE hesapla
        valid_idx = all_stacking_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, 'Actual'],
            all_stacking_preds.loc[valid_idx]
        )

        # Isınma fold'ları için basit ortalamayla doldur (rapor bütünlüğü için)
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_stacking_preds.loc[valid_idx]

        return full_preds, mape

    # ==================================================================
    # STRATEJİ 2: TERS MAPE AĞIRLIKLANDIRMA
    # ==================================================================
    def _inverse_mape_weighting(self, full_df):
        """
        Mantık:
        ───────
        Her model için önceki fold'lardaki MAPE skoru hesaplanır.
        Düşük MAPE = iyi model → daha fazla ağırlık alır.
        
        Örnek:
          XGB MAPE = %3.0  → 1/3.0 = 0.333
          LGBM MAPE = %2.0 → 1/2.0 = 0.500  ← En iyi model
          CAT MAPE = %2.5  → 1/2.5 = 0.400
          
          Normalize et (toplamı 1 yap):
          XGB: 0.333/1.233 = 0.270
          LGBM: 0.500/1.233 = 0.405  ← En fazla ağırlık
          CAT: 0.400/1.233 = 0.324
          
          Final tahmin = 0.270*XGB + 0.405*LGBM + 0.324*CAT
        
        Bu yöntem her fold için ağırlıkları günceller (adaptif).
        """
        if 'fold_id' not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df['fold_id'].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        inv_parts = []

        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue

            # Önceki fold'lardaki verileri al
            prior_mask = full_df['fold_id'].isin(fold_ids[:i])
            prior = full_df.loc[prior_mask]
            test_mask = full_df['fold_id'] == fid
            test = full_df.loc[test_mask]

            # Her modelin önceki fold'lardaki MAPE'sini hesapla
            model_mapes = []
            for col in self.PRED_COLS:
                m = calculate_mape(prior['Actual'], prior[col])
                model_mapes.append(max(m, 1e-6))  # Sıfıra bölünmeyi engelle

            # Ters MAPE ağırlıkları hesapla ve normalize et
            inv = np.array([1.0 / m for m in model_mapes])
            weights = inv / inv.sum()  # Toplamı 1 olacak şekilde normalize

            # Bu fold'un tahminini ağırlıklı ortalama olarak hesapla
            preds = sum(w * test[col].values for w, col in zip(weights, self.PRED_COLS))
            fold_preds = pd.Series(preds, index=test.index)
            inv_parts.append(fold_preds)

        all_inv_preds = pd.concat(inv_parts)

        valid_idx = all_inv_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, 'Actual'],
            all_inv_preds.loc[valid_idx]
        )

        # Final ağırlıkları hesapla ve kaydet (tüm veri üzerinden)
        final_mapes = [max(calculate_mape(full_df['Actual'], full_df[c]), 1e-6) for c in self.PRED_COLS]
        inv_final = np.array([1.0 / m for m in final_mapes])
        self.best_weights = dict(zip(self.PRED_COLS, np.round(inv_final / inv_final.sum(), 4)))
        print(f"   → Ters MAPE Ağırlıkları: {self.best_weights}")

        # Isınma fold'ları için basit ortalamayla doldur
        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_inv_preds.loc[valid_idx]

        return full_preds, mape

    # ==================================================================
    # STRATEJİ 3: KISITLI OPTİMİZASYON (scipy)
    # ==================================================================
    def _constrained_optimization(self, full_df):
        """
        Mantık:
        ───────
        scipy.optimize.minimize kullanarak en iyi ağırlıkları arar.
        
        Hedef fonksiyon: MAPE'yi minimize et
        Kısıtlar:
          1. Tüm ağırlıklar ≥ 0  (negatif ağırlık anlamsız)
          2. Ağırlıkların toplamı = 1  (normalize)
        
        Yöntem: SLSQP (Sequential Least Squares Programming)
        Bu, kısıtlı optimizasyon için yaygın kullanılan bir algoritmadır.
        
        Ters MAPE'den farkı: scipy daha esnek, doğrusal olmayan
        ilişkileri de yakalayabilir. Ama bazen Inverse MAPE kadar 
        iyi çalışmayabilir (overfitting riski biraz daha fazla).
        """
        if 'fold_id' not in full_df.columns:
            return None, None

        fold_ids = sorted(full_df['fold_id'].unique())
        if len(fold_ids) < self.MIN_WARMUP_FOLDS + 1:
            return None, None

        opt_parts = []

        for i, fid in enumerate(fold_ids):
            if i < self.MIN_WARMUP_FOLDS:
                continue

            prior_mask = full_df['fold_id'].isin(fold_ids[:i])
            prior = full_df.loc[prior_mask]
            test_mask = full_df['fold_id'] == fid
            test = full_df.loc[test_mask]

            prior_actuals = prior['Actual'].values
            prior_preds_matrix = prior[self.PRED_COLS].values  # Shape: (n, 3)
            test_preds_matrix = test[self.PRED_COLS].values

            # Hedef fonksiyon: ağırlıklar (w) verilince MAPE hesapla
            def mape_objective(w):
                blended = prior_preds_matrix @ w  # Matris çarpımı: (n,3) @ (3,) = (n,)
                return calculate_mape(prior_actuals, blended)

            # Kısıtlar ve sınırlar
            n_models = len(self.PRED_COLS)
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Toplam = 1
            bounds = [(0, 1)] * n_models  # Her ağırlık 0 ile 1 arası
            x0 = np.ones(n_models) / n_models  # Başlangıç: eşit ağırlık (1/3, 1/3, 1/3)

            # Optimizasyonu çalıştır
            res = minimize(mape_objective, x0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10})

            w_opt = res.x  # Bulunan optimal ağırlıklar
            preds = test_preds_matrix @ w_opt  # Bu fold'un tahmini
            fold_preds = pd.Series(preds, index=test.index)
            opt_parts.append(fold_preds)

        all_opt_preds = pd.concat(opt_parts)

        valid_idx = all_opt_preds.index
        mape = calculate_mape(
            full_df.loc[valid_idx, 'Actual'],
            all_opt_preds.loc[valid_idx]
        )

        # Final ağırlıkları hesapla (tüm veri üzerinden)
        prior_actuals = full_df['Actual'].values
        prior_preds_matrix = full_df[self.PRED_COLS].values

        def final_objective(w):
            return calculate_mape(prior_actuals, prior_preds_matrix @ w)

        n_models = len(self.PRED_COLS)
        res_final = minimize(final_objective, np.ones(n_models) / n_models,
                            method='SLSQP',
                            bounds=[(0, 1)] * n_models,
                            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                            options={'maxiter': 500, 'ftol': 1e-10})
        self.best_weights = dict(zip(self.PRED_COLS, np.round(res_final.x, 4)))
        print(f"   → Optimizasyon Ağırlıkları: {self.best_weights}")

        full_preds = full_df[self.PRED_COLS].mean(axis=1).copy()
        full_preds.loc[valid_idx] = all_opt_preds.loc[valid_idx]

        return full_preds, mape

    # ==================================================================
    # FİNAL ÜRETİM MODELİ
    # ==================================================================
    def _train_final_meta_model(self, full_df):
        """
        Kazanan strateji belirlendikten SONRA, üretim (production) 
        kullanımı için TÜM OOF verisi üzerinde bir Ridge modeli eğitir.
        
        Bu sızıntı DEĞİL çünkü:
        - OOS performansını yukarıda zaten kanıtladık
        - Bu model sadece gelecekteki gerçek tahminlerde kullanılacak
        - Daha fazla veriyle eğitmek = daha güvenilir katsayılar
        """
        X_meta = full_df[self.PRED_COLS]
        y_meta = full_df['Actual']
        self.meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_model.fit(X_meta, y_meta)

        coefs = dict(zip(self.PRED_COLS, np.round(self.meta_model.coef_, 4)))
        print(f"[StackingManager] Üretim Modeli Katsayıları: {coefs}")
        print(f"[StackingManager] Üretim Modeli Bias: {self.meta_model.intercept_:.4f}")

    def save_model(self, filename="stacking_ridge.joblib"):
        """Meta-modeli ve strateji bilgisini diske kaydeder."""
        if self.meta_model is None:
            print("[StackingManager] Kaydedilecek model yok.")
            return
        path = os.path.join(self.model_dir, filename)

        save_data = {
            'meta_model': self.meta_model,
            'best_method': self.best_method,
            'best_weights': self.best_weights
        }
        joblib.dump(save_data, path)
        print(f"[StackingManager] Meta-model kaydedildi: {path}")