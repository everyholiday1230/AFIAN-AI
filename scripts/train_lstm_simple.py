"""
간단한 LSTM 모델 학습
시계열 예측에 특화된 경량 모델
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

print('='*70)
print('🤖 LSTM MODEL TRAINING')
print('='*70)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n🔥 Device: {device}')

# Load data (2023만 사용 - 메모리 절약)
print('\n📥 Loading training data (2023)...')
data_dir = Path('data/historical_5min_features')
df = pd.read_parquet(data_dir / 'BTCUSDT_2023_1m.parquet')
print(f'   Loaded: {len(df):,} rows')

# Features
feature_cols = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_width', 'ATR_14',
    'returns_1', 'returns_3', 'returns_12',
    'volatility_12', 'volatility_48',
    'volume_ma_ratio', 'hour', 'day_of_week'
]

# Target
df['target'] = df['close'].pct_change(1).shift(-1) * 100
df = df.dropna()

# Prepare sequences
sequence_length = 60  # 5 hours
X_data = df[feature_cols].values
y_data = df['target'].values

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

# Create sequences
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

print(f'\n🔧 Creating sequences (length={sequence_length})...')
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
print(f'   Sequences: {len(X_seq):,}')

# Split
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print(f'   Train: {len(X_train):,}')
print(f'   Test:  {len(X_test):,}')

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

# Initialize
print(f'\n🏗️  Building LSTM model...')
input_size = len(feature_cols)
model = LSTMModel(input_size, hidden_size=32, num_layers=1, dropout=0.1).to(device)

print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 256
n_epochs = 20

# Training
print(f'\n🎓 Training ({n_epochs} epochs)...')

train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

best_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t).item()
    
    print(f'   Epoch {epoch+1:2d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}')
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # Save best model
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'   Early stopping at epoch {epoch+1}')
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluate
print(f'\n📊 Final Evaluation...')
model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).cpu().numpy()
    test_pred = model(X_test_t).cpu().numpy()

train_pred_orig = scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
test_pred_orig = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

train_rmse = np.sqrt(np.mean((train_pred_orig - y_train_orig)**2))
test_rmse = np.sqrt(np.mean((test_pred_orig - y_test_orig)**2))

print(f'   Train RMSE: {train_rmse:.4f}%')
print(f'   Test RMSE:  {test_rmse:.4f}%')

# Save model
print(f'\n💾 Saving model...')
output_dir = Path('models/lstm')
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / f'lstm_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'sequence_length': sequence_length,
    'feature_cols': feature_cols,
}, model_path)

print(f'   Saved to: {model_path}')

print(f'\n{'='*70}')
print('✅ LSTM Training completed!')
print(f'{'='*70}\n')
print('📋 Next steps:')
print('   1. Run backtest with LSTM model')
print('   2. Compare with Random Forest')
print('   3. Train on multi-year data')
