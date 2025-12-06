"""
Wavelet Transform 기반 노이즈 제거

목적: 시계열 데이터에서 노이즈를 제거하고 주요 트렌드/사이클 보존

핵심 기술:
- DWT (Discrete Wavelet Transform): 이산 웨이블릿 변환
- Soft/Hard Thresholding: 노이즈 계수 제거
- Multi-Resolution Analysis: 다중 해상도 분석

Reference:
- "Wavelet Methods for Time Series Analysis" (Percival & Walden)
- "A Practical Guide to Wavelet Analysis" (Torrence & Compo)

수학적 배경:
DWT: x(t) = Σ c_j φ_j(t) + Σ d_k ψ_k(t)
- c_j: Approximation coefficients (저주파 성분)
- d_k: Detail coefficients (고주파 성분)
- φ_j: Scaling function
- ψ_k: Wavelet function

전략적 활용:
- 가격 데이터의 노이즈 제거
- 주요 트렌드 추출
- 사이클 성분 분리
"""

import numpy as np
import pywt
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import numba


class ThresholdMethod(Enum):
    """Thresholding 방법"""
    SOFT = "soft"  # Soft thresholding (연속적)
    HARD = "hard"  # Hard thresholding (불연속적)
    GARROTE = "garrote"  # Non-negative garrote
    GREATER = "greater"  # 큰 값만 유지
    LESS = "less"  # 작은 값만 유지


@dataclass
class WaveletDenoiseResult:
    """Wavelet Denoise 결과"""
    denoised: np.ndarray  # 노이즈 제거된 신호
    noise: np.ndarray  # 추출된 노이즈
    snr_db: float  # Signal-to-Noise Ratio (dB)
    coeffs_original: List  # 원본 웨이블릿 계수
    coeffs_denoised: List  # 노이즈 제거된 계수
    threshold: float  # 사용된 임계값


class WaveletDenoiser:
    """
    Wavelet Transform 기반 노이즈 제거기
    
    특징:
    - 다양한 Wavelet Family 지원 (Daubechies, Symlets, Coiflets 등)
    - Soft/Hard Thresholding
    - 자동 임계값 계산 (VisuShrink, BayesShrink)
    - Multi-level decomposition
    
    Args:
        wavelet: Wavelet 종류 (기본: 'db8' - Daubechies 8)
        level: 분해 레벨 (기본: None - 자동 계산)
        threshold_method: Thresholding 방법
        noise_sigma: 노이즈 표준편차 (None이면 자동 추정)
    """
    
    def __init__(
        self,
        wavelet: str = 'db8',
        level: Optional[int] = None,
        threshold_method: ThresholdMethod = ThresholdMethod.SOFT,
        noise_sigma: Optional[float] = None
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method
        self.noise_sigma = noise_sigma
        
        # Wavelet 정보 검증
        if wavelet not in pywt.wavelist(kind='discrete'):
            raise ValueError(f"Invalid wavelet: {wavelet}")
    
    def denoise(
        self,
        signal: np.ndarray,
        threshold_scale: float = 1.0
    ) -> WaveletDenoiseResult:
        """
        노이즈 제거
        
        Args:
            signal: 입력 신호 (1D array)
            threshold_scale: 임계값 스케일 조정 (1.0 = 기본)
            
        Returns:
            WaveletDenoiseResult
        """
        # 레벨 자동 계산
        if self.level is None:
            level = pywt.dwt_max_level(len(signal), self.wavelet)
        else:
            level = self.level
        
        # DWT (Discrete Wavelet Transform)
        coeffs_original = pywt.wavedec(signal, self.wavelet, level=level)
        
        # 노이즈 추정
        if self.noise_sigma is None:
            # MAD (Median Absolute Deviation) 기반 추정
            detail_coeffs = coeffs_original[-1]  # 가장 고주파 성분
            sigma = self._estimate_noise_mad(detail_coeffs)
        else:
            sigma = self.noise_sigma
        
        # Thresholding
        threshold = self._calculate_threshold(
            coeffs_original,
            sigma,
            threshold_scale
        )
        
        coeffs_denoised = self._apply_threshold(
            coeffs_original,
            threshold
        )
        
        # IDWT (Inverse DWT)
        denoised = pywt.waverec(coeffs_denoised, self.wavelet)
        
        # 원본 길이와 맞추기 (패딩 제거)
        if len(denoised) > len(signal):
            denoised = denoised[:len(signal)]
        
        # 노이즈 성분
        noise = signal - denoised
        
        # SNR 계산
        snr_db = self._calculate_snr(signal, noise)
        
        return WaveletDenoiseResult(
            denoised=denoised,
            noise=noise,
            snr_db=snr_db,
            coeffs_original=coeffs_original,
            coeffs_denoised=coeffs_denoised,
            threshold=threshold
        )
    
    def _estimate_noise_mad(self, detail_coeffs: np.ndarray) -> float:
        """
        MAD (Median Absolute Deviation) 기반 노이즈 추정
        
        σ = MAD / 0.6745
        
        Reference: Donoho & Johnstone (1994)
        """
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        sigma = mad / 0.6745
        return sigma
    
    def _calculate_threshold(
        self,
        coeffs: List[np.ndarray],
        sigma: float,
        scale: float
    ) -> float:
        """
        임계값 계산 (VisuShrink)
        
        λ = σ * sqrt(2 * log(N))
        
        Reference: Donoho & Johnstone (1994)
        """
        # 신호 길이
        N = len(coeffs[0])
        
        # Universal threshold (VisuShrink)
        threshold = sigma * np.sqrt(2 * np.log(N)) * scale
        
        return threshold
    
    def _apply_threshold(
        self,
        coeffs: List[np.ndarray],
        threshold: float
    ) -> List[np.ndarray]:
        """
        Thresholding 적용
        
        Approximation coefficients (cA)는 유지하고,
        Detail coefficients (cD)에만 thresholding 적용
        """
        coeffs_denoised = [coeffs[0].copy()]  # cA (approximation) 유지
        
        # Detail coefficients에 thresholding
        for detail in coeffs[1:]:
            if self.threshold_method == ThresholdMethod.SOFT:
                denoised = pywt.threshold(detail, threshold, mode='soft')
            elif self.threshold_method == ThresholdMethod.HARD:
                denoised = pywt.threshold(detail, threshold, mode='hard')
            elif self.threshold_method == ThresholdMethod.GARROTE:
                denoised = pywt.threshold(detail, threshold, mode='garrote')
            elif self.threshold_method == ThresholdMethod.GREATER:
                denoised = pywt.threshold(detail, threshold, mode='greater')
            elif self.threshold_method == ThresholdMethod.LESS:
                denoised = pywt.threshold(detail, threshold, mode='less')
            else:
                denoised = pywt.threshold(detail, threshold, mode='soft')
            
            coeffs_denoised.append(denoised)
        
        return coeffs_denoised
    
    def _calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Signal-to-Noise Ratio (SNR) 계산
        
        SNR(dB) = 10 * log10(P_signal / P_noise)
        """
        signal_power = np.sum(signal ** 2)
        noise_power = np.sum(noise ** 2)
        
        if noise_power < 1e-10:
            return 100.0  # 매우 높은 SNR
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def decompose_signal(
        self,
        signal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        신호를 다중 주파수 성분으로 분해
        
        Returns:
            {
                'approximation': 저주파 성분 (트렌드),
                'detail_1': 고주파 성분 (레벨 1),
                'detail_2': 고주파 성분 (레벨 2),
                ...
            }
        """
        level = self.level or pywt.dwt_max_level(len(signal), self.wavelet)
        
        coeffs = pywt.wavedec(signal, self.wavelet, level=level)
        
        # 각 성분 복원
        components = {}
        
        # Approximation (트렌드)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        components['approximation'] = pywt.waverec(approx_coeffs, self.wavelet)[:len(signal)]
        
        # Details (고주파 성분)
        for i in range(1, len(coeffs)):
            detail_coeffs = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
            detail_coeffs[i] = coeffs[i]
            components[f'detail_{i}'] = pywt.waverec(detail_coeffs, self.wavelet)[:len(signal)]
        
        return components


class MultiScaleWaveletDenoiser:
    """
    다중 스케일 Wavelet 노이즈 제거
    
    여러 스케일에서 노이즈를 제거하여 더 강력한 디노이징
    """
    
    def __init__(
        self,
        wavelets: List[str] = ['db4', 'db8', 'sym8'],
        level: int = 3
    ):
        self.denoisers = [
            WaveletDenoiser(wavelet=w, level=level)
            for w in wavelets
        ]
    
    def denoise(
        self,
        signal: np.ndarray,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        다중 스케일 노이즈 제거
        
        Args:
            signal: 입력 신호
            aggregation: 결과 집계 방법 ('mean', 'median', 'weighted')
            
        Returns:
            denoised_signal
        """
        results = []
        snrs = []
        
        for denoiser in self.denoisers:
            result = denoiser.denoise(signal)
            results.append(result.denoised)
            snrs.append(result.snr_db)
        
        # 집계
        if aggregation == 'mean':
            denoised = np.mean(results, axis=0)
        elif aggregation == 'median':
            denoised = np.median(results, axis=0)
        elif aggregation == 'weighted':
            # SNR 기반 가중 평균
            weights = np.array(snrs) / np.sum(snrs)
            denoised = np.average(results, axis=0, weights=weights)
        else:
            denoised = np.mean(results, axis=0)
        
        return denoised


@numba.jit(nopython=True)
def soft_threshold_numba(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding (Numba 최적화)
    
    y = sign(x) * max(|x| - λ, 0)
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        abs_val = abs(x[i])
        if abs_val > threshold:
            result[i] = np.sign(x[i]) * (abs_val - threshold)
    return result


@numba.jit(nopython=True)
def hard_threshold_numba(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Hard thresholding (Numba 최적화)
    
    y = x if |x| > λ else 0
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        if abs(x[i]) > threshold:
            result[i] = x[i]
    return result


def adaptive_denoise_financial_series(
    prices: np.ndarray,
    volatility_window: int = 20
) -> np.ndarray:
    """
    금융 시계열 특화 적응형 노이즈 제거
    
    변동성이 높은 구간은 약한 디노이징,
    변동성이 낮은 구간은 강한 디노이징
    
    Args:
        prices: 가격 시계열
        volatility_window: 변동성 계산 윈도우
        
    Returns:
        denoised_prices
    """
    # 로그 수익률
    returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
    
    # 롤링 변동성 계산
    volatility = np.zeros_like(returns)
    for i in range(volatility_window, len(returns)):
        volatility[i] = np.std(returns[i-volatility_window:i])
    
    # 변동성이 낮은 구간과 높은 구간 분리
    median_vol = np.median(volatility[volatility > 0])
    
    # 적응형 threshold scale
    threshold_scale = np.ones_like(returns)
    threshold_scale[volatility < median_vol] = 1.5  # 강한 디노이징
    threshold_scale[volatility >= median_vol] = 0.5  # 약한 디노이징
    
    # Wavelet denoise
    denoiser = WaveletDenoiser(wavelet='db8', level=3)
    
    # 각 구간별로 다른 threshold 적용
    denoised_returns = np.zeros_like(returns)
    
    window_size = 100
    for i in range(0, len(returns), window_size):
        end_idx = min(i + window_size, len(returns))
        segment = returns[i:end_idx]
        scale = np.mean(threshold_scale[i:end_idx])
        
        result = denoiser.denoise(segment, threshold_scale=scale)
        denoised_returns[i:end_idx] = result.denoised
    
    # 가격으로 복원
    denoised_prices = prices[0] * np.exp(np.cumsum(denoised_returns))
    
    return denoised_prices


if __name__ == "__main__":
    print("🧪 Testing Wavelet Denoiser...")
    
    # 테스트 신호 생성 (트렌드 + 사이클 + 노이즈)
    np.random.seed(42)
    
    t = np.linspace(0, 10, 1000)
    trend = 0.5 * t  # 선형 트렌드
    cycle1 = 2 * np.sin(2 * np.pi * 1 * t)  # 1Hz 사이클
    cycle2 = 1 * np.sin(2 * np.pi * 5 * t)  # 5Hz 사이클
    noise = np.random.normal(0, 0.5, len(t))  # 가우시안 노이즈
    
    clean_signal = trend + cycle1 + cycle2
    noisy_signal = clean_signal + noise
    
    # 노이즈 제거 테스트
    denoiser = WaveletDenoiser(wavelet='db8', level=4, threshold_method=ThresholdMethod.SOFT)
    result = denoiser.denoise(noisy_signal)
    
    print(f"✅ Wavelet Denoising Results:")
    print(f"   - Wavelet: {denoiser.wavelet}")
    print(f"   - Level: 4")
    print(f"   - Threshold: {result.threshold:.4f}")
    print(f"   - SNR: {result.snr_db:.2f} dB")
    
    # 원본 대비 오차
    mse = np.mean((result.denoised - clean_signal) ** 2)
    print(f"   - MSE (vs clean): {mse:.4f}")
    
    # 다중 주파수 성분 분해
    print(f"\n✅ Signal Decomposition:")
    components = denoiser.decompose_signal(noisy_signal)
    for name, component in components.items():
        energy = np.sum(component ** 2)
        print(f"   - {name}: energy = {energy:.2f}")
    
    # 다중 스케일 디노이징
    print(f"\n✅ Multi-Scale Denoising:")
    multi_denoiser = MultiScaleWaveletDenoiser(wavelets=['db4', 'db8', 'sym8'], level=3)
    multi_denoised = multi_denoiser.denoise(noisy_signal, aggregation='weighted')
    multi_mse = np.mean((multi_denoised - clean_signal) ** 2)
    print(f"   - MSE (multi-scale): {multi_mse:.4f}")
    
    # 금융 시계열 테스트
    print(f"\n✅ Financial Time Series Denoising:")
    
    # 샘플 가격 시계열 (랜덤 워크 + 노이즈)
    returns = np.random.normal(0.001, 0.02, 1000)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    denoised_prices = adaptive_denoise_financial_series(prices, volatility_window=20)
    
    # 원본 vs 디노이징 비교
    original_volatility = np.std(np.diff(np.log(prices)))
    denoised_volatility = np.std(np.diff(np.log(denoised_prices)))
    
    print(f"   - Original volatility: {original_volatility*100:.2f}%")
    print(f"   - Denoised volatility: {denoised_volatility*100:.2f}%")
    print(f"   - Volatility reduction: {(1 - denoised_volatility/original_volatility)*100:.1f}%")
    
    print("\n🎉 Wavelet Denoiser test completed!")
