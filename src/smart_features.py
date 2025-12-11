import numpy as np
import pandas as pd

class SmartFeatureEngineer:
    """
    Gelişmiş özellik mühendisliği (Advanced Feature Engineering) işlemlerini yürütür.
    Özellikle 'Smart Neighbor' (En Benzer Gün) gibi karmaşık hesaplamalar burada yapılır.
    """
    def __init__(self, df):
        self.df = df

    def add_smart_neighbor_feature(self, target_col, temp_col_candidates=None):
        """
        Geçmiş günlere (Lag'lere) bakar ve sıcaklık açısından bugüne
        EN ÇOK BENZEYEN günün tüketimini 'Smart_Lag_Load' olarak ekler.
        """
        print("[SmartFeatureEngineer] 'En Benzer Gün' (Smart Lag) hesaplanıyor...")
        
        # 1. Kullanılacak Sıcaklık Sütununu Belirle
        # Eğer kullanıcı özel bir liste vermediyse veya listedekiler yoksa otomatik bul
        selected_temp_col = None
        
        # Öncelik: Aggregation ile oluşturulmuş ortalama sütunlar
        if temp_col_candidates:
            for col in temp_col_candidates:
                if col in self.df.columns:
                    selected_temp_col = col
                    break
        
        # Eğer hala bulamadıysak, içinde 'Hissedilen' geçen ilk sütunu al
        if selected_temp_col is None:
            potential_cols = [c for c in self.df.columns if 'Hissedilen' in c or 'Temperature' in c]
            if potential_cols:
                selected_temp_col = potential_cols[0]
        
        if selected_temp_col is None:
            print("   -> UYARI: Referans alınacak sıcaklık sütunu bulunamadı! Smart Lag atlanıyor.")
            return self.df

        print(f"   -> Referans Sıcaklık: {selected_temp_col}")

        # 2. Adaylar: Geçmiş 1. gün, 2. gün ... 7. gün
        # Yani model sadece düne değil, son 1 haftanın en uygun gününe bakacak.
        lags_to_check = [24, 48, 72, 96, 120, 144, 168] 
        
        # Başlangıç Değerleri
        # Varsayılan olarak Smart Lag = Lag24 (Dün) olsun.
        # Min Diff = Sonsuz (Başlangıçta hiçbir şey bilmiyoruz)
        self.df['Smart_Lag_Load'] = self.df[target_col].shift(24) 
        self.df['Min_Temp_Diff'] = 9999.0 

        # 3. Vektörel Karşılaştırma Döngüsü
        for lag in lags_to_check:
            # O geçmiş günün sıcaklığı
            past_temp = self.df[selected_temp_col].shift(lag)
            
            # O geçmiş günün tüketimi
            past_load = self.df[target_col].shift(lag)
            
            # Bugüne olan sıcaklık farkı (Mutlak değer)
            current_diff = (self.df[selected_temp_col] - past_temp).abs()
            
            # --- SEÇİM MANTIĞI ---
            # Eğer bu geçmiş günün sıcaklık farkı, şimdiye kadar bulduğumdan daha azsa;
            # Maske oluştur: Bu lag'in daha iyi olduğu satırlar
            better_match_mask = current_diff < self.df['Min_Temp_Diff']
            
            # Sadece daha iyi eşleşme bulunan satırları güncelle
            self.df.loc[better_match_mask, 'Min_Temp_Diff'] = current_diff[better_match_mask]
            self.df.loc[better_match_mask, 'Smart_Lag_Load'] = past_load[better_match_mask]
            
        # Temizlik: Geçici fark sütununu sil
        self.df.drop(columns=['Min_Temp_Diff'], inplace=True)
        
        print("   -> 'Smart_Lag_Load' özelliği başarıyla eklendi.")
        return self.df