"""
Strategist (Decision Transformer) Trainer
행동 최적화 모델 학습 (매수/매도/홀드 결정)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class StrategistDataset(Dataset):
    """Decision Transformer용 데이터셋"""
    
    def __init__(self, data_dir: str, symbols: List[str], years: List[int], context_length: int):
        self.context_length = context_length
        
        print(f'\n📥 Loading data for Strategist...')
        dfs = []
        for symbol in symbols:
            for year in years:
                file_path = Path(data_dir) / f'{symbol}_{year}_1m.parquet'
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df['symbol'] = symbol
                    
                    # Calculate rewards (returns)
                    df['reward'] = df['close'].pct_change() * 100
                    df['cumulative_return'] = (1 + df['reward']/100).cumprod()
                    
                    dfs.append(df)
                    print(f'   ✅ {symbol} {year}: {len(df):,} rows')
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f'\n✅ Total data: {len(self.data):,} rows')
        
        # Features
        self.feature_cols = [
            'close', 'volume', 'RSI_14', 'MACD', 'ATR_14',
            'returns_1', 'returns_3', 'volatility_12'
        ]
        
        self.data = self.data.dropna()
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.data[self.feature_cols] = self.scaler.fit_transform(self.data[self.feature_cols])
        
        print(f'   After preprocessing: {len(self.data):,} rows')
    
    def __len__(self):
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        # Get context window
        context = self.data.iloc[idx:idx+self.context_length]
        
        states = context[self.feature_cols].values
        rewards = context['reward'].values
        
        # Return-to-go (cumulative future rewards)
        rtg = rewards[::-1].cumsum()[::-1]
        
        # Actions (simplified: 1=buy, 0=hold, -1=sell)
        actions = np.sign(rewards)
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'rtg': torch.FloatTensor(rtg),
            'timesteps': torch.LongTensor(np.arange(self.context_length))
        }


class DecisionTransformer(pl.LightningModule):
    """Decision Transformer 모델"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        state_dim = 8  # feature_cols 개수
        action_dim = 1
        hidden_size = config['hidden_size']
        
        # Embeddings
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Linear(action_dim, hidden_size)
        self.rtg_embed = nn.Linear(1, hidden_size)
        self.timestep_embed = nn.Embedding(1000, hidden_size)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config['num_heads'],
            dim_feedforward=hidden_size * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )
        
        # Action prediction head
        self.action_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, states, actions, rtg, timesteps):
        batch_size, seq_len = states.shape[:2]
        
        # Embed
        state_emb = self.state_embed(states)
        action_emb = self.action_embed(actions.unsqueeze(-1))
        rtg_emb = self.rtg_embed(rtg.unsqueeze(-1))
        time_emb = self.timestep_embed(timesteps)
        
        # Combine: state + action + rtg + time
        x = state_emb + action_emb + rtg_emb + time_emb
        
        # Transform
        x = self.transformer(x)
        
        # Predict action
        action_pred = self.action_pred(x)
        
        return action_pred.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        actions_pred = self(
            batch['states'],
            batch['actions'],
            batch['rtg'],
            batch['timesteps']
        )
        
        loss = self.loss_fn(actions_pred, batch['actions'])
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        actions_pred = self(
            batch['states'],
            batch['actions'],
            batch['rtg'],
            batch['timesteps']
        )
        
        loss = self.loss_fn(actions_pred, batch['actions'])
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['max_epochs']
        )
        return [optimizer], [scheduler]


class StrategistTrainer:
    """Strategist 학습 관리자"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def train(self):
        print('\n🔧 Preparing Strategist datasets...')
        
        # Datasets
        train_dataset = StrategistDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=self.config['train_years'],
            context_length=self.config['context_length']
        )
        
        val_dataset = StrategistDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=[self.config['test_year']],
            context_length=self.config['context_length']
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f'   Train batches: {len(train_loader)}')
        print(f'   Val batches: {len(val_loader)}')
        
        # Model
        print('\n🏗️  Building Strategist (Decision Transformer)...')
        model = DecisionTransformer(self.config)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f'   Parameters: {n_params:,}')
        
        # Callbacks & Trainer (similar to Oracle)
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['output_dir'],
            filename='strategist-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 20),
            mode='min'
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator='gpu' if self.config.get('use_gpu') and torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=self.config.get('precision', 32),
            gradient_clip_val=self.config.get('gradient_clip_val', 1.0),
            callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor()],
            logger=TensorBoardLogger(self.config['output_dir'], name='tensorboard_logs'),
            log_every_n_steps=self.config.get('log_every_n_steps', 100),
        )
        
        print('\n🎓 Training Strategist...')
        trainer.fit(model, train_loader, val_loader)
        
        # Save best
        best_path = Path(self.config['output_dir']) / 'best_model.ckpt'
        import shutil
        shutil.copy(checkpoint_callback.best_model_path, best_path)
        
        print(f'\n✅ Strategist training completed!')
        print(f'   Best model: {best_path}')
