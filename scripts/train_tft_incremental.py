"""
TFT 점진적 학습 (메모리 효율 버전)
연도별로 나눠서 학습 후 통합
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
from pytorch_lightning.callbacks import EarlyStopping

print('='*70)
print('🤖 TFT INCREMENTAL TRAINING (Year by Year)')
print('='*70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\n🔥 Device: {device}')

data_dir = Path('data/historical_5min_features')

# 1년치씩 학습
for year in [2021, 2022, 2023]:
    print(f'\n{'='*70}')
    print(f'📅 Training on {year} data')
    print(f'{'='*70}')
    
    # Load single year
    print(f'\n📥 Loading {year} data...')
    df = pd.read_parquet(data_dir / f'BTCUSDT_{year}_1m.parquet')
    print(f'   Rows: {len(df):,}')
    
    # Prepare data
    df = df.reset_index(drop=True)
    df['time_idx'] = range(len(df))
    df['group'] = 'BTCUSDT'
    
    # Target
    df['target'] = df['close'].pct_change(1).shift(-1) * 100
    df = df.dropna(subset=['target'])
    
    print(f'   After preprocessing: {len(df):,} rows')
    
    # Features
    time_varying_unknown_reals = [
        'close', 'volume',
        'SMA_10', 'EMA_12', 'RSI_14', 'MACD',
        'returns_1', 'volatility_12',
    ]
    
    time_varying_known_reals = ['hour', 'day_of_week']
    
    # Split
    split_idx = int(len(df) * 0.8)
    
    # Create dataset
    print(f'\n🔧 Creating TimeSeriesDataSet...')
    max_encoder_length = 30  # 2.5 hours (30 * 5min) - 줄임
    max_prediction_length = 6  # 30 min ahead - 줄임
    
    training = TimeSeriesDataSet(
        df[:split_idx],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(
        training, df[split_idx:], predict=True, stop_randomization=True
    )
    
    print(f'   Train samples: {len(training):,}')
    print(f'   Val samples: {len(validation):,}')
    
    # Dataloaders
    batch_size = 64  # 줄임
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )
    
    print(f'   Batch size: {batch_size}')
    
    # Build model
    print(f'\n🏗️  Building TFT (lightweight)...')
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=16,  # 매우 작게 (32->16)
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,  # 작게 (8->4)
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    
    n_params = sum(p.numel() for p in tft.parameters())
    print(f'   Parameters: {n_params:,}')
    
    # Train
    print(f'\n🎓 Training (max 5 epochs, limited batches)...')
    
    trainer = pl.Trainer(
        max_epochs=5,  # 짧게
        accelerator="cpu",
        enable_model_summary=False,
        gradient_clip_val=0.1,
        limit_train_batches=20,  # 배치 제한 (메모리 절약)
        limit_val_batches=5,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=2, mode="min"),
        ],
        logger=False,
        enable_progress_bar=True,
    )
    
    try:
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Save
        output_dir = Path('models/tft')
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f'tft_{year}.ckpt'
        trainer.save_checkpoint(model_path)
        
        print(f'\n✅ {year} model saved: {model_path}')
        
    except Exception as e:
        print(f'\n❌ Error training {year}: {e}')
        print('   Skipping...')
        continue
    
    # Clean up
    del tft, trainer, training, validation
    del train_dataloader, val_dataloader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f'\n{'='*70}')
print('✅ TFT INCREMENTAL TRAINING COMPLETED!')
print(f'{'='*70}\n')

print('📋 Trained models:')
for year in [2021, 2022, 2023]:
    model_path = Path(f'models/tft/tft_{year}.ckpt')
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f'   {year}: {model_path} ({size_mb:.2f} MB)')

print('\n📋 Next steps:')
print('   1. Load best year model (likely 2023)')
print('   2. Run backtest on 2024 data')
print('   3. Compare with Random Forest')
print('   4. Ensemble multiple year models')
