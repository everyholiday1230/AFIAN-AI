#!/usr/bin/env python3
"""
🚀 QUANTUM ALPHA V2 - ULTIMATE SOLUTION
Claude + Gemini + GPT 통합 전략 구현

Phase 1: 즉시 실행 가능
- Multi-Timeframe Strategy
- Minimal Feature Set (자체 지표 + 핵심 5개)
- Strong Risk Management
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime


class MultiTimeframeStrategy:
    """멀티 타임프레임 전략"""
    
    def __init__(self):
        pass
    
    def resample_to_4h(self, df_5min):
        """5분봉 → 4시간봉"""
        df = df_5min.copy()
        df.index = pd.to_datetime(df.index)
        
        df_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 4시간봉 지표
        df_4h['SMA_50'] = df_4h['close'].rolling(50).mean()
        df_4h['SMA_200'] = df_4h['close'].rolling(200).mean()
        df_4h['ATR_14'] = self.calculate_atr(df_4h, 14)
        
        return df_4h
    
    def resample_to_1h(self, df_5min):
        """5분봉 → 1시간봉"""
        df = df_5min.copy()
        df.index = pd.to_datetime(df.index)
        
        df_1h = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 1시간봉 지표 (2배 확대)
        df_1h['RSI_21'] = self.calculate_rsi(df_1h['close'], 21)
        df_1h['MACD'], df_1h['MACD_signal'], df_1h['MACD_hist'] = self.calculate_macd(df_1h['close'], 24, 52, 18)
        df_1h['BB_upper'], df_1h['BB_middle'], df_1h['BB_lower'] = self.calculate_bollinger_bands(df_1h['close'], 20)
        
        return df_1h
    
    def resample_to_15min(self, df_5min):
        """5분봉 → 15분봉"""
        df = df_5min.copy()
        df.index = pd.to_datetime(df.index)
        
        df_15min = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 15분봉 지표
        df_15min['RSI_14'] = self.calculate_rsi(df_15min['close'], 14)
        df_15min['volume_ma'] = df_15min['volume'].rolling(20).mean()
        
        return df_15min
    
    @staticmethod
    def calculate_rsi(prices, period):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std=2):
        """볼린저 밴드 계산"""
        middle = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def calculate_atr(df, period=14):
        """ATR 계산"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def get_trend_signal(self, df_4h):
        """4시간봉 트렌드 신호"""
        if df_4h.empty or len(df_4h) < 200:
            return 0
        
        latest = df_4h.iloc[-1]
        
        # SMA 50 vs SMA 200
        if latest['SMA_50'] > latest['SMA_200']:
            return 1  # 상승 트렌드
        elif latest['SMA_50'] < latest['SMA_200']:
            return -1  # 하락 트렌드
        else:
            return 0  # 횡보
    
    def get_entry_signal(self, df_1h, trend):
        """1시간봉 진입 신호"""
        if df_1h.empty or len(df_1h) < 50:
            return 0
        
        latest = df_1h.iloc[-1]
        
        # 트렌드 방향에만 거래
        if trend == 0:
            return 0
        
        # RSI + MACD 조합
        rsi_signal = 0
        if latest['RSI_21'] < 30:  # 과매도
            rsi_signal = 1
        elif latest['RSI_21'] > 70:  # 과매수
            rsi_signal = -1
        
        macd_signal = 0
        if latest['MACD_hist'] > 0:
            macd_signal = 1
        elif latest['MACD_hist'] < 0:
            macd_signal = -1
        
        # 트렌드 방향과 일치할 때만
        combined_signal = (rsi_signal + macd_signal) / 2
        
        if trend > 0 and combined_signal > 0:
            return combined_signal
        elif trend < 0 and combined_signal < 0:
            return combined_signal
        else:
            return 0
    
    def get_execution_timing(self, df_15min):
        """15분봉 실행 타이밍"""
        if df_15min.empty or len(df_15min) < 20:
            return 1.0
        
        latest = df_15min.iloc[-1]
        
        # 거래량 확인
        volume_ratio = latest['volume'] / latest['volume_ma'] if latest['volume_ma'] > 0 else 1.0
        
        # RSI 미세 조정
        rsi_factor = 1.0
        if latest['RSI_14'] < 40:
            rsi_factor = 1.2  # 강한 매수
        elif latest['RSI_14'] > 60:
            rsi_factor = 0.8  # 약한 매수
        
        return volume_ratio * rsi_factor


class RiskManager:
    """리스크 관리"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # 리스크 파라미터
        self.stop_loss_pct = 0.02      # 2%
        self.trailing_stop_pct = 0.015  # 1.5%
        self.max_position_pct = 0.3     # 30%
        self.risk_per_trade = 0.01      # 1%
    
    def calculate_position_size(self, signal, volatility, capital):
        """포지션 크기 계산"""
        
        # Volatility 기반 동적 레버리지
        if volatility > 0.05:
            leverage = 1.0
        elif volatility > 0.03:
            leverage = 1.5
        else:
            leverage = 2.0
        
        # Kelly Criterion (보수적)
        win_rate = 0.55
        avg_win = 0.015
        avg_loss = 0.01
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_conservative = kelly * 0.5  # 50% Kelly
        
        # 최종 포지션
        position_value = capital * kelly_conservative * leverage * abs(signal)
        max_value = capital * self.max_position_pct
        
        return min(position_value, max_value), leverage
    
    def check_stop_loss(self, entry_price, current_price, position_type):
        """Stop Loss 체크"""
        if position_type == 'long':
            loss_pct = (current_price - entry_price) / entry_price
        else:  # short
            loss_pct = (entry_price - current_price) / entry_price
        
        if loss_pct < -self.stop_loss_pct:
            return True, f"Stop Loss Hit: {loss_pct*100:.2f}%"
        
        return False, "OK"


class QuantumAlphaV2Backtester:
    """V2 백테스터"""
    
    def __init__(self, data_path, custom_features=None):
        self.data_path = data_path
        self.custom_features = custom_features or []
        
        self.mtf = MultiTimeframeStrategy()
        self.risk_mgr = RiskManager()
    
    def load_data(self):
        """데이터 로드"""
        df = pd.read_parquet(self.data_path)
        
        # timestamp를 인덱스로
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def prepare_multi_timeframe(self, df_5min):
        """멀티 타임프레임 준비"""
        print("📊 Resampling to multiple timeframes...")
        
        df_4h = self.mtf.resample_to_4h(df_5min)
        df_1h = self.mtf.resample_to_1h(df_5min)
        df_15min = self.mtf.resample_to_15min(df_5min)
        
        print(f"   4H bars: {len(df_4h)}")
        print(f"   1H bars: {len(df_1h)}")
        print(f"   15min bars: {len(df_15min)}")
        
        return df_4h, df_1h, df_15min
    
    def run_backtest(self, year=2024):
        """백테스트 실행"""
        print("\n" + "="*70)
        print(f"🚀 QUANTUM ALPHA V2 - ULTIMATE BACKTEST ({year})")
        print("="*70)
        
        # 데이터 로드
        df_5min = self.load_data()
        print(f"\n✅ Data loaded: {len(df_5min):,} rows")
        
        # 멀티 타임프레임 준비
        df_4h, df_1h, df_15min = self.prepare_multi_timeframe(df_5min)
        
        # 백테스트 초기화
        capital = 10000.0
        position = 0.0
        entry_price = 0.0
        position_type = None
        
        trades = []
        equity = [capital]
        
        # 15분봉 기준으로 거래
        for i in range(len(df_15min)):
            timestamp = df_15min.index[i]
            current_price = df_15min['close'].iloc[i]
            
            # 현재 시점의 4H, 1H 데이터 가져오기
            df_4h_current = df_4h[df_4h.index <= timestamp]
            df_1h_current = df_1h[df_1h.index <= timestamp]
            df_15min_current = df_15min.iloc[:i+1]
            
            # 충분한 데이터가 있을 때만
            if len(df_4h_current) < 200 or len(df_1h_current) < 50:
                equity.append(capital)
                continue
            
            # 1. 4H 트렌드
            trend = self.mtf.get_trend_signal(df_4h_current)
            
            # 2. 1H 진입 신호
            entry_signal = self.mtf.get_entry_signal(df_1h_current, trend)
            
            # 3. 15min 실행 타이밍
            execution_timing = self.mtf.get_execution_timing(df_15min_current)
            
            # 최종 신호
            final_signal = entry_signal * execution_timing
            
            # Stop Loss 체크
            if position != 0:
                stop_hit, msg = self.risk_mgr.check_stop_loss(
                    entry_price, current_price, position_type
                )
                
                if stop_hit:
                    # 포지션 청산
                    pnl = position * (current_price - entry_price)
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'close',
                        'reason': msg,
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital
                    })
                    
                    position = 0
                    entry_price = 0
                    position_type = None
            
            # 신호가 강할 때만 거래
            if abs(final_signal) > 0.5:
                # 변동성 계산
                volatility = df_15min_current['close'].pct_change().std()
                
                # 포지션 크기 계산
                position_value, leverage = self.risk_mgr.calculate_position_size(
                    final_signal, volatility, capital
                )
                
                # 기존 포지션 청산
                if position != 0:
                    pnl = position * (current_price - entry_price)
                    capital += pnl
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'close',
                        'reason': 'signal_change',
                        'price': current_price,
                        'pnl': pnl,
                        'capital': capital
                    })
                
                # 신규 포지션
                desired_position = (position_value / current_price) * np.sign(final_signal)
                
                # 수수료 차감
                commission = abs(desired_position * current_price) * 0.0004
                capital -= commission
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'open',
                    'signal': final_signal,
                    'price': current_price,
                    'position': desired_position,
                    'leverage': leverage,
                    'capital': capital
                })
                
                position = desired_position
                entry_price = current_price
                position_type = 'long' if position > 0 else 'short'
            
            # 자본 업데이트
            unrealized = position * (current_price - entry_price) if position != 0 else 0
            equity.append(capital + unrealized)
        
        # 메트릭 계산
        self.calculate_metrics(equity, trades, year)
        
        return equity, trades
    
    def calculate_metrics(self, equity, trades, year):
        """메트릭 계산 및 출력"""
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = drawdowns.max() * 100
        
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 96)) if returns.std() > 0 else 0
        
        # 승률 계산
        winning_trades = [t for t in trades if t['type'] == 'close' and t.get('pnl', 0) > 0]
        total_close_trades = [t for t in trades if t['type'] == 'close']
        win_rate = (len(winning_trades) / len(total_close_trades) * 100) if total_close_trades else 0
        
        # 결과 출력
        print(f"\n{'='*70}")
        print("📈 BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"{'Strategy:':<25} Multi-Timeframe V2")
        print(f"{'Year:':<25} {year}")
        print(f"\n{'Initial Capital:':<25} ${10000:,.2f}")
        print(f"{'Final Capital:':<25} ${equity[-1]:,.2f}")
        print(f"{'Total Return:':<25} {total_return:+.2f}%")
        print(f"\n{'Max Drawdown:':<25} {max_dd:.2f}%")
        print(f"{'Sharpe Ratio:':<25} {sharpe:.2f}")
        print(f"{'Win Rate:':<25} {win_rate:.2f}%")
        print(f"{'Total Trades:':<25} {len(total_close_trades):,}")
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
        results_file = results_dir / f"ultimate_v2_backtest_{year}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"QUANTUM ALPHA V2 - Ultimate Backtest Results ({year})\n")
            f.write("="*70 + "\n\n")
            f.write("Strategy: Multi-Timeframe (4H + 1H + 15min)\n")
            f.write("Features: Minimal Set (10-12 indicators)\n")
            f.write("Risk Management: Stop Loss + Position Sizing\n\n")
            f.write(f"Total Return:     {total_return:+.2f}%\n")
            f.write(f"Max Drawdown:     {max_dd:.2f}%\n")
            f.write(f"Sharpe Ratio:     {sharpe:.2f}\n")
            f.write(f"Win Rate:         {win_rate:.2f}%\n")
            f.write(f"Total Trades:     {len(total_close_trades):,}\n\n")
            f.write(f"Initial Capital:  ${10000:,.2f}\n")
            f.write(f"Final Capital:    ${equity[-1]:,.2f}\n")
        
        print(f"\n💾 Results saved: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Quantum Alpha V2 - Ultimate Solution')
    parser.add_argument('--year', type=int, default=2024, help='Year to backtest')
    parser.add_argument('--data-path', type=str, 
                       default='data/historical_5min_features/BTCUSDT_2024_1m.parquet',
                       help='Path to data file')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 QUANTUM ALPHA V2 - ULTIMATE SOLUTION")
    print("="*70)
    print("\n📊 Strategy Components:")
    print("   ✅ Multi-Timeframe: 4H + 1H + 15min")
    print("   ✅ Indicator Scaling: 2x (RSI 21, MACD 24/52)")
    print("   ✅ Risk Management: Stop Loss + Position Sizing")
    print("   ✅ Dynamic Leverage: 1.0x - 2.0x")
    
    backtester = QuantumAlphaV2Backtester(args.data_path)
    equity, trades = backtester.run_backtest(args.year)
    
    print("\n✅ Backtest complete!")
    print("\n🎯 Next Steps:")
    print("   1. Review detailed results in results/")
    print("   2. Compare with v1: python scripts/compare_strategies.py")
    print("   3. Walk-Forward validation: python scripts/walk_forward.py")


if __name__ == "__main__":
    main()
