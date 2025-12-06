"""
점진적 학습: 연도별로 순차 학습
메모리 절약 버전
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

print('='*70)
print('🤖 INCREMENTAL MODEL TRAINING (2019-2023)')
print('='*70)

data_dir = Path('data/historical_5min_features')

feature_cols = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_width', 'ATR_14',
    'returns_1', 'returns_3', 'returns_12',
    'volatility_12', 'volatility_48',
    'volume_ma_ratio', 'hour', 'day_of_week'
]

# Train on 2021-2023 (최근 3년만)
train_files = [
    'BTCUSDT_2021_1m.parquet',
    'BTCUSDT_2022_1m.parquet',
    'BTCUSDT_2023_1m.parquet',
]

print('\n📥 Loading training data (2021-2023)...')
X_parts = []
y_parts = []

for f in train_files:
    df = pd.read_parquet(data_dir / f)
    df['target'] = df['close'].pct_change(1).shift(-1) * 100
    df = df.dropna(subset=feature_cols + ['target'])
    
    X_parts.append(df[feature_cols].values)
    y_parts.append(df['target'].values)
    print(f'   {f}: {len(df):,} rows')

X_train = np.vstack(X_parts)
y_train = np.concatenate(y_parts)

print(f'\n✅ Total training samples: {len(X_train):,}')

# Load test data
print('\n📥 Loading test data (2024)...')
df_test = pd.read_parquet(data_dir / 'BTCUSDT_2024_1m.parquet')
df_test['target'] = df_test['close'].pct_change(1).shift(-1) * 100
df_test = df_test.dropna(subset=feature_cols + ['target'])

X_test = df_test[feature_cols].values
y_test = df_test['target'].values
print(f'   Test samples: {len(X_test):,}')

# Train
print('\n🌲 Training Random Forest (50 trees)...')
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    n_jobs=4,
    random_state=42,
    verbose=2
)

model.fit(X_train, y_train)

# Evaluate
print('\n📊 Evaluating...')
train_pred = model.predict(X_train[:10000])  # Sample만
test_pred = model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

print(f'\n{'='*70}')
print('📈 RESULTS')
print(f'{'='*70}')
print(f'\nTraining: 2021-2023 (3 years, {len(X_train):,} samples)')
print(f'Testing:  2024 ({len(X_test):,} samples)')
print(f'\nTest Metrics (2024):')
print(f'   RMSE: {test_rmse:.4f}%')
print(f'   R²:   {test_r2:.4f}')

# Feature importance
print(f'\n🔝 Top 10 Features:')
feature_importance = dict(zip(feature_cols, model.feature_importances_))
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_features[:10]:
    print(f'   {feat:20s}: {imp:.4f}')

# Save
output_dir = Path('models/full')
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / f'rf_2021_2023_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
joblib.dump(model, model_path)

print(f'\n💾 Model saved: {model_path}')
print(f'\n{'='*70}')
print('✅ Training completed!')
print(f'{'='*70}\n')
