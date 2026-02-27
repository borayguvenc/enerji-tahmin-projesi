import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

# Proje kök dizini
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# ---------------------------------------------------------
class TabularModel(nn.Module):
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        super(TabularModel, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # Stabilite için şart
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) # Overfit engellemek için
            in_dim = h_dim
            
        # Çıkış Katmanı (Regresyon olduğu için 1 nöron)
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
class ANNManager:
    def __init__(self, epochs=100, batch_size=64, learning_rate=0.001, 
                 hidden_layers=[256, 128, 64], dropout_rate=0.3):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers # <--- Yeni
        self.dropout_rate = dropout_rate   # <--- Yeni
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        print(f"[ANNManager] Cihaz: {self.device}")

    def train_model(self, X_train, y_train, X_test, y_test, sample_weight=None):
        """
        Modeli eğitir. Sample Weight destekler.
        """
        print("[ANN] Veriler ölçekleniyor (Scaling)...")
        # 1. Feature Scaling (Ağaçlar istemez ama ANN buna muhtaçtır)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 2. Tensor Dönüşümü
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.FloatTensor(y_train.values).view(-1, 1).to(self.device)
        X_test_t = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_t = torch.FloatTensor(y_test.values).view(-1, 1).to(self.device)
        
        # Ağırlık Varsa Tensor Yap
        weights_t = None
        if sample_weight is not None:
            weights_t = torch.FloatTensor(sample_weight).view(-1, 1).to(self.device)

        # 3. Model Başlatma
        input_dim = X_train.shape[1]
        self.model = TabularModel(
            input_dim, 
            hidden_layers=self.hidden_layers, 
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Optimizer & Loss
        criterion = nn.L1Loss(reduction='none') # MAE Loss (Reduction none yapıyoruz ki ağırlık çarpabilelim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning Rate Scheduler (Plato çizerse LR düşür)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 4. Eğitim Döngüsü (Training Loop)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        # Eğer ağırlık varsa dataset'e ekle
        if weights_t is not None:
            train_dataset = TensorDataset(X_train_t, y_train_t, weights_t)
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_limit = 20 # 20 epoch boyunca iyileşme olmazsa dur
        
        print(f"[ANN] Eğitim Başlıyor ({self.epochs} Epoch)...")
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                if weights_t is not None:
                    # Ağırlıklı Eğitim
                    batch_X, batch_y, batch_w = batch
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss = (loss * batch_w).mean() # Ağırlıkla çarp ve ortalama al
                else:
                    # Normal Eğitim
                    batch_X, batch_y = batch
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y).mean()
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_t)
                val_loss = nn.L1Loss()(val_outputs, y_test_t).item()
            
            # Scheduler Step
            scheduler.step(val_loss)
            
            # Early Stopping Kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi modeli kaydet (Hafızada)
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_limit:
                print(f"🛑 Early Stopping! Epoch: {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # En iyi ağırlıkları yükle
        self.model.load_state_dict(best_model_state)
        print(f"✅ Eğitim Bitti. Best Val Loss: {best_val_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        # Scale etmeyi unutma!
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_t)
            
        return preds.cpu().numpy().flatten()

    def save_model(self, filename="ann_model.pth"):
        path = os.path.join(project_root, 'Model', filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler # Scaler'ı da kaydetmek zorundayız!
        }, path)
        print(f"[ANN] Model kaydedildi: {path}")

    def load_model(self, filename="ann_model.pth"):
        path = os.path.join(project_root, 'Model', filename)
        checkpoint = torch.load(path)
        
        # Model yapısını kurmamız lazım (Input dim scaler'dan anlaşılabilir)
        self.scaler = checkpoint['scaler']
        input_dim = self.scaler.n_features_in_
        
        self.model = TabularModel(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()