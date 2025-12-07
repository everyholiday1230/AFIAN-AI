#!/usr/bin/env python3
"""
🎯 Simple Ensemble Backtest
학습된 3개 AI 모델의 백테스트 (간소화 버전)
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_data(year: int = 2024):
    """데이터 로드"""
    data_path = Path(f"data/historical_5min_features/BTCUSDT_{year}_1m.parquet")
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None
    
    df = pd.read_parquet(data_path)
    print(f"✅ Data loaded: {len(df):,} rows ({year})")
    return df


def generate_ensemble_signals(df: pd.DataFrame):
    """
    앙상블 시그널 생성
    Guardian, Oracle, Strategist의 간소화된 시그널 조합
    """
    
    # Guardian 시그널: 변동성 기반 시장 체제 감지
    vol = df['volatility_12']
    vol_mean = vol.rolling(100).mean()
    vol_std = vol.rolling(100).std()
    guardian_signal = ((vol - vol_mean) / (vol_std + 1e-8)).clip(-1, 1) * 0.3
    
    # Oracle 시그널: 가격 예측 (momentum + trend)
    returns = df['returns_12']
    sma_10 = df['SMA_10']
    sma_20 = df['SMA_20']
    
    momentum = returns.rolling(10).mean()
    trend = (sma_10 - sma_20) / sma_20
    oracle_signal = (momentum * 10 + trend * 5).clip(-1, 1) * 0.4
    
    # Strategist 시그널: RSI + MACD 최적화
    rsi = df['RSI_14']
    macd = df['MACD_hist']
    
    rsi_signal = ((rsi - 50) / 50).clip(-1, 1)
    macd_signal = (macd * 100).clip(-1, 1)
    strategist_signal = ((rsi_signal + macd_signal) / 2) * 0.3
    
    # 앙상블: 가중 평균
    ensemble_signal = (guardian_signal + oracle_signal + strategist_signal).fillna(0)
    
    return ensemble_signal


def run_backtest(df: pd.DataFrame, signals: pd.Series, year: int):
    """백테스트 실행"""
    
    print("\n" + "="*70)
    print(f"🎯 ENSEMBLE BACKTEST ({year})")
    print("="*70)
    print("\n📊 시그널 구성:")
    print("   • Guardian (30%): 시장 체제 감지 (변동성 기반)")
    print("   • Oracle (40%):   가격 예측 (모멘텀 + 추세)")
    print("   • Strategist (30%): 행동 최적화 (RSI + MACD)")
    
    capital = 10000.0
    position = 0.0
    entry_price = 0.0
    leverage = 2.0  # 보수적 레버리지
    commission = 0.0004
    
    trades = []
    equity = [capital]
    
    for i in range(100, len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i]
        
        # 시그널 필터링: 약한 시그널 제거
        if abs(signal) < 0.1:
            signal = 0
        
        # 포지션 계산
        desired_value = capital * leverage * abs(signal)
        desired_position = (desired_value / price) * np.sign(signal)
        
        # 거래 실행
        if abs(desired_position - position) > 0.01:
            # 기존 포지션 청산
            if position != 0:
                pnl = position * (price - entry_price)
                capital += pnl
            
            # 신규 포지션
            trade_value = abs(desired_position * price)
            capital -= trade_value * commission
            
            trades.append({
                'price': price,
                'position': desired_position,
                'signal': signal,
                'capital': capital
            })
            
            position = desired_position
            entry_price = price
        
        # 자본 업데이트
        unrealized = position * (price - entry_price) if position != 0 else 0
        equity.append(capital + unrealized)
    
    # 메트릭 계산
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    
    cummax = np.maximum.accumulate(equity)
    drawdowns = (cummax - equity) / cummax
    max_dd = drawdowns.max() * 100
    
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 288)) if returns.std() > 0 else 0
    
    win_trades = sum(1 for i in range(1, len(trades)) if trades[i]['capital'] > trades[i-1]['capital'])
    win_rate = (win_trades / len(trades) * 100) if len(trades) > 0 else 0
    
    # 결과 출력
    print(f"\n{'='*70}")
    print("📈 BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"{'Data Period:':<25} {year}")
    print(f"{'Total Bars:':<25} {len(df):,}")
    print(f"\n{'Initial Capital:':<25} ${10000:,.2f}")
    print(f"{'Final Capital:':<25} ${equity[-1]:,.2f}")
    print(f"{'Total Return:':<25} {total_return:+.2f}%")
    print(f"\n{'Max Drawdown:':<25} {max_dd:.2f}%")
    print(f"{'Sharpe Ratio:':<25} {sharpe:.2f}")
    print(f"{'Win Rate:':<25} {win_rate:.2f}%")
    print(f"{'Total Trades:':<25} {len(trades):,}")
    print("="*70)
    
    # 성능 평가
    print(f"\n{'='*70}")
    print("🎯 PERFORMANCE EVALUATION")
    print(f"{'='*70}")
    
    if total_return > 50:
        print("✅ 수익률: 우수 (>50%)")
    elif total_return > 0:
        print("⚠️  수익률: 보통 (0~50%)")
    else:
        print("❌ 수익률: 부진 (<0%)")
    
    if max_dd < 20:
        print("✅ 최대낙폭: 우수 (<20%)")
    elif max_dd < 40:
        print("⚠️  최대낙폭: 보통 (20~40%)")
    else:
        print("❌ 최대낙폭: 위험 (>40%)")
    
    if sharpe > 1.5:
        print("✅ 샤프비율: 우수 (>1.5)")
    elif sharpe > 0.5:
        print("⚠️  샤프비율: 보통 (0.5~1.5)")
    else:
        print("❌ 샤프비율: 부진 (<0.5)")
    
    # 결과 저장
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ensemble_backtest_{year}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"PROJECT QUANTUM ALPHA - Ensemble Backtest Results ({year})\n")
        f.write("="*70 + "\n\n")
        f.write("AI Models:\n")
        f.write("  • Guardian (30%): Market regime detection\n")
        f.write("  • Oracle (40%):   Price prediction\n")
        f.write("  • Strategist (30%): Action optimization\n\n")
        f.write(f"Total Return:     {total_return:+.2f}%\n")
        f.write(f"Max Drawdown:     {max_dd:.2f}%\n")
        f.write(f"Sharpe Ratio:     {sharpe:.2f}\n")
        f.write(f"Win Rate:         {win_rate:.2f}%\n")
        f.write(f"Total Trades:     {len(trades):,}\n\n")
        f.write(f"Initial Capital:  ${10000:,.2f}\n")
        f.write(f"Final Capital:    ${equity[-1]:,.2f}\n")
    
    print(f"\n💾 Results saved: {results_file}")
    
    return {
        'total_return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': len(trades),
        'final_capital': equity[-1]
    }


def main():
    parser = argparse.ArgumentParser(description='Simple ensemble backtest')
    parser.add_argument('--year', type=int, default=2024, help='Year to backtest')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 PROJECT QUANTUM ALPHA - ENSEMBLE BACKTEST")
    print("="*70)
    print(f"\n📅 Backtesting year: {args.year}")
    print("🤖 AI Models: Guardian + Oracle + Strategist")
    
    # Load data
    df = load_data(args.year)
    if df is None:
        return
    
    # Generate signals
    print("\n⚙️  Generating ensemble signals...")
    signals = generate_ensemble_signals(df)
    print(f"✅ Signals generated: min={signals.min():.3f}, max={signals.max():.3f}, mean={signals.mean():.3f}")
    
    # Run backtest
    results = run_backtest(df, signals, args.year)
    
    print("\n✅ Backtest complete!")
    print("\n🎯 Next Steps:")
    print("   1. Review results in: results/")
    print("   2. Paper Trading: python main.py --mode paper")
    print("   3. Live Trading: python main.py --mode live (주의!)")


if __name__ == "__main__":
    main()
