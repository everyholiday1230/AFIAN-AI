"""
전체 데이터(2019-2023)로 Random Forest 학습
2024년으로 Out-of-Sample 테스트
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime

print('='*70)
print('🤖 FULL DATA MODEL TRAINING (2019-2023)')
print('='*70)

# Load all training data (2019-2023)
data_dir = Path('data/historical_5min_features')
train_files = [
    'BTCUSDT_2019_1m.parquet',
    'BTCUSDT_2020_1m.parquet',
    'BTCUSDT_2021_1m.parquet',
    'BTCUSDT_2022_1m.parquet',
    'BTCUSDT_2023_1m.parquet',
]

print('\n📥 Loading training data...')
train_dfs = []
for f in train_files:
    df = pd.read_parquet(data_dir / f)
    train_dfs.append(df)
    print(f'   {f}: {len(df):,} rows')

df_train = pd.concat(train_dfs, ignore_index=True)
print(f'\n✅ Total training data: {len(df_train):,} rows')

# Load test data (2024)
print('\n📥 Loading test data...')
df_test = pd.read_parquet(data_dir / 'BTCUSDT_2024_1m.parquet')
print(f'   BTCUSDT_2024_1m.parquet: {len(df_test):,} rows')

# Prepare features
feature_cols = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_width', 'ATR_14',
    'returns_1', 'returns_3', 'returns_12',
    'volatility_12', 'volatility_48',
    'volume_ma_ratio', 'hour', 'day_of_week'
]

print('\n🔧 Preparing features...')

# Target: next 5min return
df_train['target'] = df_train['close'].pct_change(1).shift(-1) * 100
df_test['target'] = df_test['close'].pct_change(1).shift(-1) * 100

# Drop NaN
df_train = df_train.dropna(subset=feature_cols + ['target'])
df_test = df_test.dropna(subset=feature_cols + ['target'])

X_train = df_train[feature_cols]
y_train = df_train['target']
X_test = df_test[feature_cols]
y_test = df_test['target']

print(f'   Train samples: {len(X_train):,}')
print(f'   Test samples:  {len(X_test):,}')
print(f'   Features: {len(feature_cols)}')

# Train model
print('\n🌲 Training Random Forest...')
print('   (This may take 5-10 minutes...)')

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

# Predictions
print('\n📊 Evaluating...')
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f'\n{'='*70}')
print('📈 TRAINING RESULTS')
print(f'{'='*70}')
print(f'\nTrain Metrics (2019-2023):')
print(f'   RMSE: {train_rmse:.4f}%')
print(f'   MAE:  {train_mae:.4f}%')
print(f'   R²:   {train_r2:.4f}')
print(f'\nTest Metrics (2024 - Out-of-Sample):')
print(f'   RMSE: {test_rmse:.4f}%')
print(f'   MAE:  {test_mae:.4f}%')
print(f'   R²:   {test_r2:.4f}')

# Feature importance
print(f'\n🔝 Top 10 Important Features:')
feature_importance = dict(zip(feature_cols, model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_features[:10]:
    print(f'   {feat:20s}: {imp:.4f}')

# Save model
output_dir = Path('models/full')
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / f'rf_full_2019_2023_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
joblib.dump(model, model_path)

print(f'\n💾 Model saved: {model_path}')
print(f'\n{'='*70}')
print('✅ Training completed!')
print(f'{'='*70}\n')
