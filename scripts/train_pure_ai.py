#!/usr/bin/env python3
"""
🤖 Pure AI Learning System - 룰 없이 100% AI 자율 학습

AI가 스스로:
- RSI의 의미 발견
- MACD 조합 학습  
- 최적 진입/청산 타이밍 학습
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class TradingDataset(Dataset):
    """순수 AI 학습용 데이터셋"""
    
    def __init__(self, data_path, lookback=60):
        self.lookback = lookback
        
        # 데이터 로드
        df = pd.read_parquet(data_path)
        
        # 특징: Raw 지표 값 (룰 없음!)
        self.feature_cols = [
            'close', 'volume',
            'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'ATR_14',
            'returns_1', 'returns_3', 'returns_12',
            'volatility_12', 'volatility_48',
            'volume_ma_ratio',
            'hour', 'day_of_week'
        ]
        
        # 타겟: 미래 수익률 (24개 봉 후 = 2시간)
        df['target'] = df['close'].pct_change(24).shift(-24) * 100
        
        # NaN 제거
        df = df.dropna()
        
        # 정규화
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        df[self.feature_cols] = self.scaler_X.fit_transform(df[self.feature_cols])
        df[['target']] = self.scaler_y.fit_transform(df[['target']])
        
        self.data = df
        print(f"✅ Dataset: {len(df):,} samples")
    
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        # Lookback 윈도우
        window = self.data.iloc[idx:idx+self.lookback]
        features = window[self.feature_cols].values
        
        # 타겟
        target = self.data.iloc[idx+self.lookback]['target']
        
        return {
            'features': torch.FloatTensor(features),
            'target': torch.FloatTensor([target])
        }


class PureAITransformer(nn.Module):
    """
    순수 AI Transformer
    - 룰 없음
    - AI가 모든 패턴 학습
    """
    
    def __init__(self, input_dim=22, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, d_model]
        
        # Take last timestep
        x = x[:, -1, :]  # [batch, d_model]
        
        # Output
        output = self.output_head(x)  # [batch, 1]
        
        return output


class PureAITrainer:
    """순수 AI 학습 트레이너"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss
        self.criterion = nn.MSELoss()
    
    def train_epoch(self):
        """1 epoch 학습"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=100):
        """전체 학습"""
        print("\n" + "="*70)
        print("🤖 Pure AI Training Started (No Rules!)")
        print("="*70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Print
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), 'models/pure_ai/best_model.pth')
                print(f"   ✅ Best model saved (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    break
        
        print("\n✅ Training complete!")


def backtest_pure_ai(model, data_path, scaler_X, scaler_y, device='cuda'):
    """순수 AI 백테스트"""
    print("\n" + "="*70)
    print("🎯 Pure AI Backtest (No Rules!)")
    print("="*70)
    
    # 데이터 로드
    df = pd.read_parquet(data_path)
    
    # 특징 준비
    feature_cols = [
        'close', 'volume',
        'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
        'ATR_14',
        'returns_1', 'returns_3', 'returns_12',
        'volatility_12', 'volatility_48',
        'volume_ma_ratio',
        'hour', 'day_of_week'
    ]
    
    df_features = df[feature_cols].fillna(0)
    df_features_scaled = scaler_X.transform(df_features)
    
    # 백테스트 초기화
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    trades = []
    equity = [capital]
    
    lookback = 60
    model.eval()
    
    with torch.no_grad():
        for i in range(lookback, len(df)):
            # 현재 가격
            current_price = df['close'].iloc[i]
            
            # AI 예측
            window = torch.FloatTensor(df_features_scaled[i-lookback:i]).unsqueeze(0).to(device)
            prediction = model(window).cpu().item()
            
            # Denormalize
            prediction = scaler_y.inverse_transform([[prediction]])[0][0]
            
            # 신호 변환 (-1 ~ 1)
            signal = np.tanh(prediction / 2.0)  # 2% 예측 → 신호 1.0
            
            # 포지션 계산
            leverage = 2.0
            desired_value = capital * leverage * abs(signal)
            desired_position = (desired_value / current_price) * np.sign(signal)
            
            # 거래 실행 (신호 임계값 0.3)
            if abs(signal) > 0.3 and abs(desired_position - position) > 0.01:
                # 청산
                if position != 0:
                    pnl = position * (current_price - entry_price)
                    capital += pnl
                
                # 수수료
                commission = abs(desired_position * current_price) * 0.0004
                capital -= commission
                
                trades.append({
                    'price': current_price,
                    'signal': signal,
                    'prediction': prediction,
                    'capital': capital
                })
                
                position = desired_position
                entry_price = current_price
            
            # 자본 업데이트
            unrealized = position * (current_price - entry_price) if position != 0 else 0
            equity.append(capital + unrealized)
    
    # 메트릭 계산
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    cummax = np.maximum.accumulate(equity)
    max_dd = ((cummax - equity) / cummax).max() * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 288)) if returns.std() > 0 else 0
    
    win_trades = sum(1 for i in range(1, len(trades)) if trades[i]['capital'] > trades[i-1]['capital'])
    win_rate = (win_trades / len(trades) * 100) if trades else 0
    
    # 결과 출력
    print(f"\n{'='*70}")
    print("📈 RESULTS")
    print(f"{'='*70}")
    print(f"{'Strategy:':<25} Pure AI (No Rules)")
    print(f"{'Model:':<25} Transformer")
    print(f"\n{'Initial Capital:':<25} ${10000:,.2f}")
    print(f"{'Final Capital:':<25} ${equity[-1]:,.2f}")
    print(f"{'Total Return:':<25} {total_return:+.2f}%")
    print(f"\n{'Max Drawdown:':<25} {max_dd:.2f}%")
    print(f"{'Sharpe Ratio:':<25} {sharpe:.2f}")
    print(f"{'Win Rate:':<25} {win_rate:.2f}%")
    print(f"{'Total Trades:':<25} {len(trades):,}")
    print("="*70)
    
    # AI가 발견한 패턴 분석
    print(f"\n{'='*70}")
    print("🧠 AI Discovered Patterns")
    print(f"{'='*70}")
    
    if trades:
        signals = [t['signal'] for t in trades]
        predictions = [t['prediction'] for t in trades]
        
        print(f"{'Signal Range:':<25} [{min(signals):.3f}, {max(signals):.3f}]")
        print(f"{'Prediction Range:':<25} [{min(predictions):.3f}%, {max(predictions):.3f}%]")
        print(f"{'Avg Prediction:':<25} {np.mean(predictions):.3f}%")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': len(trades)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, 
                       default='data/historical_5min_features/BTCUSDT_2023_1m.parquet')
    parser.add_argument('--test-data', type=str,
                       default='data/historical_5min_features/BTCUSDT_2024_1m.parquet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 PURE AI LEARNING SYSTEM")
    print("="*70)
    print("\n📋 Configuration:")
    print(f"   Train Data: {args.train_data}")
    print(f"   Test Data: {args.test_data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Device: {args.device}")
    
    # 디렉토리 생성
    Path("models/pure_ai").mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 준비
    print("\n📊 Preparing datasets...")
    train_dataset = TradingDataset(args.train_data, lookback=60)
    val_dataset = TradingDataset(args.test_data, lookback=60)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 생성
    model = PureAITransformer(
        input_dim=22,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    )
    
    print(f"\n🤖 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 학습
    trainer = PureAITrainer(model, train_loader, val_loader, device=args.device)
    trainer.train(epochs=args.epochs)
    
    # 백테스트
    model.load_state_dict(torch.load('models/pure_ai/best_model.pth'))
    results = backtest_pure_ai(
        model, 
        args.test_data,
        train_dataset.scaler_X,
        train_dataset.scaler_y,
        device=args.device
    )
    
    print("\n✅ Complete!")


if __name__ == "__main__":
    main()
