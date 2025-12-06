"""
Guardian (Contrastive VAE) Trainer
시장 체제 감지 모델 학습 (Bull/Bear/Sideways)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class GuardianDataset(Dataset):
    """시장 체제 감지용 데이터셋"""
    
    def __init__(self, data_dir: str, symbols: List[str], years: List[int], window_size: int):
        self.window_size = window_size
        
        print(f'\n📥 Loading data for Guardian...')
        dfs = []
        for symbol in symbols:
            for year in years:
                file_path = Path(data_dir) / f'{symbol}_{year}_1m.parquet'
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df['symbol'] = symbol
                    
                    # Calculate regime labels (simple heuristic)
                    df['returns_window'] = df['close'].pct_change(window_size)
                    df['volatility_window'] = df['close'].pct_change().rolling(window_size).std()
                    
                    # Bull: returns > 2%, Bear: returns < -2%, Sideways: between
                    df['regime'] = 1  # Sideways
                    df.loc[df['returns_window'] > 0.02, 'regime'] = 2  # Bull
                    df.loc[df['returns_window'] < -0.02, 'regime'] = 0  # Bear
                    
                    dfs.append(df)
                    print(f'   ✅ {symbol} {year}: {len(df):,} rows')
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f'\n✅ Total data: {len(self.data):,} rows')
        
        # Features
        self.feature_cols = [
            'close', 'volume', 'returns_1', 'returns_3', 'returns_12',
            'volatility_12', 'volatility_48', 'RSI_14', 'MACD', 'ATR_14'
        ]
        
        self.data = self.data.dropna()
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.data[self.feature_cols] = self.scaler.fit_transform(self.data[self.feature_cols])
        
        print(f'   After preprocessing: {len(self.data):,} rows')
        print(f'   Regime distribution:')
        print(f'      Bear (0):     {(self.data["regime"]==0).sum():,}')
        print(f'      Sideways (1): {(self.data["regime"]==1).sum():,}')
        print(f'      Bull (2):     {(self.data["regime"]==2).sum():,}')
    
    def __len__(self):
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx):
        # Get window
        window = self.data.iloc[idx:idx+self.window_size]
        
        features = window[self.feature_cols].values
        regime = window['regime'].iloc[-1]  # Last regime in window
        
        return {
            'features': torch.FloatTensor(features),
            'regime': torch.LongTensor([regime])
        }


class ContrastiveVAE(pl.LightningModule):
    """Contrastive VAE 모델"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        input_dim = 10  # feature_cols 개수
        window_size = config['window_size']
        latent_dim = config['latent_dim']
        hidden_dims = config['hidden_dims']
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim * window_size
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim * window_size))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Classifier (regime prediction)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, 3)  # 3 regimes
        )
        
        self.beta = config['beta']  # VAE beta
        self.temperature = config['temperature']  # Contrastive temperature
    
    def encode(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        h = self.encoder(x_flat)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x_flat = self.decoder(z)
        return x_flat.reshape(z.shape[0], self.config['window_size'], -1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        regime_logits = self.classifier(z)
        return x_recon, mu, logvar, regime_logits
    
    def vae_loss(self, x, x_recon, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss
    
    def contrastive_loss(self, z):
        # Simple contrastive loss (InfoNCE)
        z_norm = F.normalize(z, dim=1)
        similarity = torch.matmul(z_norm, z_norm.T) / self.temperature
        
        # Positive pairs: consecutive samples
        batch_size = z.shape[0]
        labels = torch.arange(batch_size, device=z.device)
        
        loss = F.cross_entropy(similarity, labels)
        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch['features']
        regime = batch['regime'].squeeze()
        
        x_recon, mu, logvar, regime_logits = self(x)
        
        # VAE loss
        vae_loss = self.vae_loss(x, x_recon, mu, logvar)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(mu)
        
        # Classification loss
        clf_loss = F.cross_entropy(regime_logits, regime)
        
        # Total loss
        loss = vae_loss + 0.1 * contrastive_loss + clf_loss
        
        # Accuracy
        regime_pred = regime_logits.argmax(dim=1)
        acc = (regime_pred == regime).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        regime = batch['regime'].squeeze()
        
        x_recon, mu, logvar, regime_logits = self(x)
        
        vae_loss = self.vae_loss(x, x_recon, mu, logvar)
        contrastive_loss = self.contrastive_loss(mu)
        clf_loss = F.cross_entropy(regime_logits, regime)
        
        loss = vae_loss + 0.1 * contrastive_loss + clf_loss
        
        regime_pred = regime_logits.argmax(dim=1)
        acc = (regime_pred == regime).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


class GuardianTrainer:
    """Guardian 학습 관리자"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def train(self):
        print('\n🔧 Preparing Guardian datasets...')
        
        # Datasets
        train_dataset = GuardianDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=self.config['train_years'],
            window_size=self.config['window_size']
        )
        
        val_dataset = GuardianDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=[self.config['test_year']],
            window_size=self.config['window_size']
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
        print('\n🏗️  Building Guardian (Contrastive VAE)...')
        model = ContrastiveVAE(self.config)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f'   Parameters: {n_params:,}')
        
        # Callbacks & Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from pytorch_lightning.loggers import TensorBoardLogger
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['output_dir'],
            filename='guardian-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 10),
            mode='min'
        )
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator='gpu' if self.config.get('use_gpu') and torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=self.config.get('precision', 32),
            gradient_clip_val=self.config.get('gradient_clip_val', 0.5),
            callbacks=[checkpoint_callback, early_stop_callback, LearningRateMonitor()],
            logger=TensorBoardLogger(self.config['output_dir'], name='tensorboard_logs'),
            log_every_n_steps=self.config.get('log_every_n_steps', 50),
        )
        
        print('\n🎓 Training Guardian...')
        trainer.fit(model, train_loader, val_loader)
        
        # Save best
        best_path = Path(self.config['output_dir']) / 'best_model.ckpt'
        import shutil
        shutil.copy(checkpoint_callback.best_model_path, best_path)
        
        print(f'\n✅ Guardian training completed!')
        print(f'   Best model: {best_path}')
