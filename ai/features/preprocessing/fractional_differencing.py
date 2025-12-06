"""
Fractional Differencing

분수 차분: 메모리 보존 + 정상성 확보

수학적 정의:
Δ^d X_t = Σ(k=0 to ∞) w_k * X_(t-k)

여기서 w_k = (-1)^k * Γ(d+1) / (Γ(k+1) * Γ(d-k+1))

Reference: "Advances in Financial Machine Learning" by Marcos Lopez de Prado
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import special
import numba


@numba.jit(nopython=True)
def _compute_weights_fast(d: float, max_lag: int, threshold: float) -> np.ndarray:
    """
    분수 차분 가중치 계산 (JIT 최적화)
    
    Args:
        d: 차분 차수 (0 < d < 1)
        max_lag: 최대 lag
        threshold: 가중치 임계값
        
    Returns:
        weights: 분수 차분 가중치
    """
    weights = np.zeros(max_lag)
    weights[0] = 1.0
    
    for k in range(1, max_lag):
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
        if abs(weights[k]) < threshold:
            return weights[:k+1]
    
    return weights


class FractionalDifferencing:
    """
    분수 차분 (Fractional Differencing)
    
    장점:
    - 시계열 정상성 확보
    - 장기 메모리 보존
    - 과도한 차분으로 인한 정보 손실 방지
    
    Args:
        d: 차분 차수 (0 < d < 1, 보통 0.5 정도)
        threshold: 가중치 임계값 (작은 가중치 무시)
    """
    
    def __init__(self, d: float = 0.5, threshold: float = 1e-5):
        if not 0 < d < 1:
            raise ValueError("d must be between 0 and 1")
        
        self.d = d
        self.threshold = threshold
        self.weights = None
    
    def compute_weights(self, max_lag: int) -> np.ndarray:
        """분수 차분 가중치 계산"""
        return _compute_weights_fast(self.d, max_lag, self.threshold)
    
    def fit_transform(
        self, 
        series: Union[pd.Series, np.ndarray],
        max_lag: Optional[int] = None
    ) -> pd.Series:
        """
        시계열 분수 차분 적용
        
        Args:
            series: 입력 시계열
            max_lag: 최대 lag (None이면 자동 결정)
            
        Returns:
            diff_series: 분수 차분된 시계열
        """
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        
        if max_lag is None:
            max_lag = len(series)
        
        # 가중치 계산
        self.weights = self.compute_weights(max_lag)
        
        # 컨볼루션을 통한 효율적 계산
        diff_values = np.convolve(series.values, self.weights, mode='valid')
        
        # 인덱스 정렬
        start_idx = len(self.weights) - 1
        new_index = series.index[start_idx:]
        
        result = pd.Series(
            diff_values, 
            index=new_index, 
            name=f'{series.name}_fracdiff_{self.d:.2f}' if series.name else None
        )
        
        return result
    
    def get_optimal_d(
        self,
        series: Union[pd.Series, np.ndarray],
        d_range: tuple = (0.0, 1.0),
        step: float = 0.01,
        significance: float = 0.05
    ) -> float:
        """
        최적 차분 차수 d 찾기 (ADF 테스트 기반)
        
        Args:
            series: 입력 시계열
            d_range: d의 탐색 범위
            step: 탐색 스텝
            significance: 유의수준
            
        Returns:
            optimal_d: 최적 차분 차수
        """
        from statsmodels.tsa.stattools import adfuller
        
        if isinstance(series, pd.Series):
            series = series.values
        
        d_values = np.arange(d_range[0], d_range[1] + step, step)
        adf_stats = []
        
        for d in d_values:
            if d == 0:
                # No differencing
                test_series = series
            else:
                # Apply fractional differencing
                temp_fd = FractionalDifferencing(d=d, threshold=self.threshold)
                test_series = temp_fd.fit_transform(pd.Series(series)).values
            
            # ADF test
            adf_result = adfuller(test_series, maxlag=1, regression='c', autolag=None)
            adf_stat = adf_result[0]
            adf_stats.append(adf_stat)
            
            # If stationary, use this d
            if adf_result[1] < significance:
                return d
        
        # If no d makes it stationary, return d with best (most negative) ADF stat
        optimal_idx = np.argmin(adf_stats)
        return d_values[optimal_idx]
    
    def plot_weights(self, max_lag: int = 50):
        """가중치 시각화"""
        import matplotlib.pyplot as plt
        
        weights = self.compute_weights(max_lag)
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(weights)), weights, alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('Weight')
        plt.title(f'Fractional Differencing Weights (d={self.d})')
        plt.grid(True, alpha=0.3)
        plt.show()


def get_memory_preserving_d(
    series: Union[pd.Series, np.ndarray],
    target_correlation: float = 0.95
) -> float:
    """
    메모리 보존 목표에 따른 최적 d 계산
    
    원본과의 상관관계가 target_correlation 이상이 되는 최대 d
    
    Args:
        series: 입력 시계열
        target_correlation: 목표 상관계수
        
    Returns:
        optimal_d: 최적 차분 차수
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    for d in np.arange(0.1, 1.0, 0.05):
        fd = FractionalDifferencing(d=d)
        diff_series = fd.fit_transform(series)
        
        # 공통 인덱스에서 상관계수 계산
        common_idx = series.index.intersection(diff_series.index)
        if len(common_idx) < 10:
            continue
        
        corr = series.loc[common_idx].corr(diff_series.loc[common_idx])
        
        if corr < target_correlation:
            return max(0.1, d - 0.05)
    
    return 0.9


if __name__ == "__main__":
    print("🧪 Testing Fractional Differencing...")
    
    # 생성: 추세 + 노이즈
    np.random.seed(42)
    n = 1000
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 5
    series = pd.Series(trend + noise, name='price')
    
    # 분수 차분 적용
    fd = FractionalDifferencing(d=0.5)
    diff_series = fd.fit_transform(series)
    
    print(f"✅ Original series length: {len(series)}")
    print(f"✅ Differenced series length: {len(diff_series)}")
    print(f"✅ Number of weights: {len(fd.weights)}")
    print(f"✅ Correlation: {series.loc[diff_series.index].corr(diff_series):.4f}")
    
    # 최적 d 찾기
    optimal_d = fd.get_optimal_d(series)
    print(f"✅ Optimal d (ADF test): {optimal_d:.2f}")
    
    # 메모리 보존 d
    memory_d = get_memory_preserving_d(series, target_correlation=0.95)
    print(f"✅ Memory-preserving d (corr > 0.95): {memory_d:.2f}")
    
    # 시각화
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 원본 vs 차분
        axes[0].plot(series.index, series.values, label='Original', alpha=0.7)
        axes[0].plot(diff_series.index, diff_series.values, label=f'Frac Diff (d={fd.d})', alpha=0.7)
        axes[0].set_title('Fractional Differencing Example')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 가중치
        weights = fd.weights[:50]
        axes[1].bar(range(len(weights)), weights, alpha=0.7)
        axes[1].set_title('Fractional Differencing Weights')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Weight')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/user/webapp/docs/fractional_differencing_example.png', dpi=150)
        print("✅ Plot saved to docs/fractional_differencing_example.png")
    except Exception as e:
        print(f"⚠️  Plotting skipped: {e}")
    
    print("\n🎉 Fractional Differencing test completed!")
