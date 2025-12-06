"""
간단한 TFT (Temporal Fusion Transformer) 학습
PyTorch Forecasting 라이브러리 사용
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

print('='*70)
print('🤖 TFT MODEL TRAINING (Temporal Fusion Transformer)')
print('='*70)

# Check PyTorch
print(f'\n🔥 PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
print(f'   Device: {"cuda" if torch.cuda.is_available() else "cpu"}')

# Load data (2021-2023)
print('\n📥 Loading training data (2021-2023)...')
data_dir = Path('data/historical_5min_features')

dfs = []
for year in [2021, 2022, 2023]:
    df = pd.read_parquet(data_dir / f'BTCUSDT_{year}_1m.parquet')
    df['year'] = year
    dfs.append(df)
    print(f'   {year}: {len(df):,} rows')

df = pd.concat(dfs, ignore_index=True)
print(f'\n✅ Total: {len(df):,} rows')

# Prepare for TFT
print('\n🔧 Preparing TimeSeriesDataSet...')

# Add time index
df['time_idx'] = range(len(df))
df['group'] = 'BTCUSDT'  # Group identifier

# Select features
static_categoricals = []
static_reals = []
time_varying_known_categoricals = []
time_varying_known_reals = ['hour', 'day_of_week']
time_varying_unknown_categoricals = []
time_varying_unknown_reals = [
    'close', 'volume',
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_width', 'ATR_14',
    'returns_1', 'returns_3',
    'volatility_12',
]

# Target: next price change
df['target'] = df['close'].pct_change(1).shift(-1) * 100
df = df.dropna(subset=['target'])

print(f'   After preprocessing: {len(df):,} rows')

# Split train/val (80/20)
max_prediction_length = 12  # Predict 1 hour ahead (12 * 5min)
max_encoder_length = 60     # Use 5 hours history (60 * 5min)

split_idx = int(len(df) * 0.8)
training = TimeSeriesDataSet(
    df[:split_idx],
    time_idx="time_idx",
    target="target",
    group_ids=["group"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=static_categoricals,
    static_reals=static_reals,
    time_varying_known_categoricals=time_varying_known_categoricals,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_categoricals=time_varying_unknown_categoricals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df[split_idx:], predict=True, stop_randomization=True)

print(f'   Training samples: {len(training):,}')
print(f'   Validation samples: {len(validation):,}')

# Create dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

print(f'   Batch size: {batch_size}')
print(f'   Train batches: {len(train_dataloader)}')
print(f'   Val batches: {len(val_dataloader)}')

# Configure TFT model
print('\n🏗️  Building TFT model...')

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=32,  # Small for faster training
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

print(f'   Model parameters: {sum(p.numel() for p in tft.parameters()):,}')

# Configure trainer
print('\n🎓 Training...')

trainer = pl.Trainer(
    max_epochs=10,  # Short for testing
    accelerator="cpu",  # Use CPU (GPU if available)
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # Limit for faster training
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    ],
    logger=False,  # Disable logging for simplicity
)

# Train
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Save model
print('\n💾 Saving model...')
output_dir = Path('models/tft')
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / 'tft_simple.ckpt'

trainer.save_checkpoint(model_path)
print(f'   Saved to: {model_path}')

# Evaluate
print('\n📊 Evaluation on validation set...')
best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

print(f'\n{'='*70}')
print('✅ TFT Training completed!')
print(f'{'='*70}\n')
print('📋 Next steps:')
print('   1. Run backtest with TFT model')
print('   2. Compare with Random Forest')
print('   3. Optimize hyperparameters')
print('   4. Train on full 5-year data')
