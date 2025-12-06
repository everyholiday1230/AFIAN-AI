"""
Advanced Technical Indicators
포괄적인 기술적 지표 라이브러리

특징:
- All major technical indicators
- Vectorized computation with Numba
- Customizable parameters
- Edge case handling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import numba
from scipy import signal


@numba.jit(nopython=True)
def _sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (Numba optimized)"""
    result = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1:i + 1])
    return result


@numba.jit(nopython=True)
def _ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (Numba optimized)"""
    result = np.full(len(prices), np.nan)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    result[period - 1] = np.mean(prices[:period])
    
    # Calculate EMA
    for i in range(period, len(prices)):
        result[i] = (prices[i] - result[i - 1]) * multiplier + result[i - 1]
    
    return result


class TechnicalIndicators:
    """
    포괄적인 기술적 지표 계산
    
    Categories:
    - Trend indicators (MA, MACD, ADX)
    - Momentum indicators (RSI, Stochastic, CCI)
    - Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
    - Volume indicators (OBV, MFI, VWAP)
    """
    
    @staticmethod
    def sma(prices: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return pd.Series(
            _sma_numba(prices.values, period),
            index=prices.index,
            name=f'SMA_{period}'
        )
    
    @staticmethod
    def ema(prices: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        # Use pandas ewm for better NaN handling
        return prices.ewm(span=period, adjust=False).mean().rename(f'EMA_{period}')
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.rename(f'RSI_{period}')
    
    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Returns:
            macd_line, signal_line, histogram
        """
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return (
            macd_line.rename('MACD'),
            signal_line.rename('MACD_signal'),
            histogram.rename('MACD_hist')
        )
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Returns:
            upper_band, middle_band, lower_band
        """
        middle = TechnicalIndicators.sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return (
            upper.rename('BB_upper'),
            middle.rename('BB_middle'),
            lower.rename('BB_lower')
        )
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range
        
        Measures market volatility
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.rename(f'ATR_{period}')
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = SMA of %K
        
        Returns:
            %K, %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return (
            k.rename(f'Stoch_%K_{k_period}'),
            d.rename(f'Stoch_%D_{d_period}')
        )
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average Directional Index
        
        Measures trend strength (0-100)
        """
        # True Range
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        # Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()
        
        pos_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        neg_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        pos_dm = pd.Series(pos_dm, index=high.index)
        neg_dm = pd.Series(neg_dm, index=low.index)
        
        # Smoothed True Range and Directional Movements
        atr_smooth = tr.rolling(window=period).mean()
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr_smooth)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr_smooth)
        
        # ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.rename(f'ADX_{period}')
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Relates volume to price change
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv.rename('OBV')
    
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index
        
        Volume-weighted RSI
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        # Money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return mfi.rename(f'MFI_{period}')
    
    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price
        
        VWAP = Σ(Price * Volume) / Σ(Volume)
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap.rename('VWAP')
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index
        
        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        """
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        
        return cci.rename(f'CCI_{period}')
    
    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels
        
        Similar to Bollinger Bands but uses ATR
        
        Returns:
            upper, middle, lower
        """
        middle = TechnicalIndicators.ema(close, period)
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)
        
        return (
            upper.rename('KC_upper'),
            middle.rename('KC_middle'),
            lower.rename('KC_lower')
        )
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R
        
        Momentum indicator (0 to -100)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return wr.rename(f'Williams_%R_{period}')
    
    @staticmethod
    def ichimoku_cloud(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud
        
        Japanese trend-following system
        
        Returns dictionary with:
            - tenkan_sen (conversion line)
            - kijun_sen (base line)
            - senkou_span_a (leading span A)
            - senkou_span_b (leading span B)
            - chikou_span (lagging span)
        """
        # Conversion Line (Tenkan-sen)
        conversion_high = high.rolling(window=conversion_period).max()
        conversion_low = low.rolling(window=conversion_period).min()
        tenkan_sen = (conversion_high + conversion_low) / 2
        
        # Base Line (Kijun-sen)
        base_high = high.rolling(window=base_period).max()
        base_low = low.rolling(window=base_period).min()
        kijun_sen = (base_high + base_low) / 2
        
        # Leading Span A (Senkou Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        span_b_high = high.rolling(window=span_b_period).max()
        span_b_low = low.rolling(window=span_b_period).min()
        senkou_span_b = ((span_b_high + span_b_low) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        chikou_span = close.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen.rename('Ichimoku_tenkan'),
            'kijun_sen': kijun_sen.rename('Ichimoku_kijun'),
            'senkou_span_a': senkou_span_a.rename('Ichimoku_senkou_a'),
            'senkou_span_b': senkou_span_b.rename('Ichimoku_senkou_b'),
            'chikou_span': chikou_span.rename('Ichimoku_chikou'),
        }


def calculate_all_indicators(
    data: pd.DataFrame,
    include_advanced: bool = True
) -> pd.DataFrame:
    """
    모든 기술적 지표 한번에 계산
    
    Args:
        data: OHLCV DataFrame
        include_advanced: 고급 지표 포함 여부
        
    Returns:
        원본 데이터 + 모든 지표
    """
    result = data.copy()
    
    # Trend indicators
    result['SMA_20'] = TechnicalIndicators.sma(data['close'], 20)
    result['SMA_50'] = TechnicalIndicators.sma(data['close'], 50)
    result['EMA_12'] = TechnicalIndicators.ema(data['close'], 12)
    result['EMA_26'] = TechnicalIndicators.ema(data['close'], 26)
    
    macd, macd_signal, macd_hist = TechnicalIndicators.macd(data['close'])
    result['MACD'] = macd
    result['MACD_signal'] = macd_signal
    result['MACD_hist'] = macd_hist
    
    # Momentum indicators
    result['RSI_14'] = TechnicalIndicators.rsi(data['close'], 14)
    
    stoch_k, stoch_d = TechnicalIndicators.stochastic(
        data['high'], data['low'], data['close']
    )
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d
    
    # Volatility indicators
    bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['close'])
    result['BB_upper'] = bb_upper
    result['BB_middle'] = bb_middle
    result['BB_lower'] = bb_lower
    
    result['ATR_14'] = TechnicalIndicators.atr(
        data['high'], data['low'], data['close'], 14
    )
    
    # Volume indicators
    result['OBV'] = TechnicalIndicators.obv(data['close'], data['volume'])
    
    if include_advanced:
        result['ADX_14'] = TechnicalIndicators.adx(
            data['high'], data['low'], data['close'], 14
        )
        
        result['MFI_14'] = TechnicalIndicators.mfi(
            data['high'], data['low'], data['close'], data['volume'], 14
        )
        
        result['VWAP'] = TechnicalIndicators.vwap(
            data['high'], data['low'], data['close'], data['volume']
        )
        
        result['CCI_20'] = TechnicalIndicators.cci(
            data['high'], data['low'], data['close'], 20
        )
        
        kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(
            data['high'], data['low'], data['close']
        )
        result['KC_upper'] = kc_upper
        result['KC_middle'] = kc_middle
        result['KC_lower'] = kc_lower
    
    return result


if __name__ == "__main__":
    from loguru import logger
    
    logger.info("🧪 Testing Technical Indicators...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
    
    data = pd.DataFrame({
        'high': np.random.randn(len(dates)).cumsum() + 40100,
        'low': np.random.randn(len(dates)).cumsum() + 39900,
        'close': np.random.randn(len(dates)).cumsum() + 40000,
        'volume': np.random.rand(len(dates)) * 1000000,
    }, index=dates)
    
    # Calculate indicators
    result = calculate_all_indicators(data, include_advanced=True)
    
    logger.info(f"✅ Calculated {len(result.columns) - len(data.columns)} indicators")
    logger.info(f"   Columns: {list(result.columns)}")
    
    # Test individual indicators
    logger.info("\n📊 Sample values:")
    logger.info(f"   RSI: {result['RSI_14'].iloc[-1]:.2f}")
    logger.info(f"   MACD: {result['MACD'].iloc[-1]:.2f}")
    logger.info(f"   ATR: {result['ATR_14'].iloc[-1]:.2f}")
    
    logger.success("✅ Technical Indicators test completed!")
