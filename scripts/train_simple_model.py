"""
간단한 ML 모델 학습 (테스트용)
5분봉 데이터로 가격 예측 모델 학습
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import argparse
from datetime import datetime


def load_data(data_dir: Path):
    """하나의 파일만 로드 (메모리 절약)"""
    # BTCUSDT 2023 데이터만 사용 (최근 데이터)
    file = data_dir / "BTCUSDT_2023_1m.parquet"
    
    if not file.exists():
        print(f"   File not found: {file}")
        print(f"   Available files:")
        for f in sorted(data_dir.glob("*.parquet")):
            print(f"     - {f.name}")
        raise FileNotFoundError(f"Cannot find {file}")
    
    print(f"   Loading: {file.name}")
    df = pd.read_parquet(file)
    return df


def prepare_features(df: pd.DataFrame):
    """Feature와 target 분리"""
    
    # Target: 다음 5분 가격 변화 (%)
    df['target'] = df['close'].pct_change(1).shift(-1) * 100  # Next 5min return
    
    # Feature 선택 (NaN 없는 것들)
    feature_cols = [
        'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'RSI_14', 'MACD', 'MACD_hist',
        'BB_width', 'ATR_14',
        'returns_1', 'returns_3', 'returns_12',
        'volatility_12', 'volatility_48',
        'volume_ma_ratio', 'hour', 'day_of_week'
    ]
    
    # Drop rows with NaN
    df = df.dropna(subset=feature_cols + ['target'])
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, df


def train_model(X_train, y_train, X_test, y_test):
    """Random Forest 학습"""
    
    print("\n🌲 Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=50,  # 절반으로 줄임
        max_depth=8,      # 더 얕게
        min_samples_split=50,
        min_samples_leaf=20,
        n_jobs=4,         # CPU 코어 제한
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/historical_5min_features')
    parser.add_argument('--output-dir', type=str, default='models/simple')
    parser.add_argument('--test-size', type=float, default=0.2)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("🤖 SIMPLE ML MODEL TRAINING")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"📥 Loading data from {data_dir}...")
    df = load_data(data_dir)
    print(f"   Loaded: {len(df):,} rows")
    
    # Prepare features
    print(f"\n🔧 Preparing features...")
    X, y, df_clean = prepare_features(df)
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(X):,}")
    print(f"   Target range: [{y.min():.2f}%, {y.max():.2f}%]")
    
    # Split data (time-based)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    
    print(f"\n{'='*70}")
    print("📈 TRAINING RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Train Metrics:")
    print(f"   RMSE: {metrics['train_rmse']:.4f}%")
    print(f"   MAE:  {metrics['train_mae']:.4f}%")
    print(f"   R²:   {metrics['train_r2']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"   RMSE: {metrics['test_rmse']:.4f}%")
    print(f"   MAE:  {metrics['test_mae']:.4f}%")
    print(f"   R²:   {metrics['test_r2']:.4f}")
    
    # Feature importance
    print(f"\n🔝 Top 10 Important Features:")
    sorted_features = sorted(
        metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, imp in sorted_features[:10]:
        print(f"   {feat:20s}: {imp:.4f}")
    
    # Save model
    model_path = output_dir / f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(model, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Training Metrics\n")
        f.write(f"================\n\n")
        f.write(f"Train RMSE: {metrics['train_rmse']:.4f}%\n")
        f.write(f"Train MAE:  {metrics['train_mae']:.4f}%\n")
        f.write(f"Train R²:   {metrics['train_r2']:.4f}\n\n")
        f.write(f"Test RMSE:  {metrics['test_rmse']:.4f}%\n")
        f.write(f"Test MAE:   {metrics['test_mae']:.4f}%\n")
        f.write(f"Test R²:    {metrics['test_r2']:.4f}\n\n")
        f.write(f"Top Features:\n")
        for feat, imp in sorted_features[:20]:
            f.write(f"  {feat}: {imp:.4f}\n")
    
    print(f"📊 Metrics saved: {metrics_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
