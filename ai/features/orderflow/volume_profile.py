"""
Volume Profile - 가격대별 거래량 분석

목적: 가격대별 유동성 분포를 분석하여 주요 지지/저항 레벨 식별

핵심 지표:
- POC (Point of Control): 최대 거래량 가격대
- VAH (Value Area High): 거래량 상위 70% 구간의 최고가
- VAL (Value Area Low): 거래량 상위 70% 구간의 최저가
- HVN (High Volume Node): 고거래량 구간
- LVN (Low Volume Node): 저거래량 구간

Reference:
- "Mind Over Markets" (James Dalton)
- "Markets in Profile" (James Dalton & Robert Dalton)

전략적 활용:
- POC는 강력한 지지/저항선으로 작용
- VAH/VAL 돌파 시 추세 전환 신호
- LVN은 빠른 가격 이동 구간
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numba


@dataclass
class VolumeProfileResult:
    """Volume Profile 결과"""
    poc: float  # Point of Control
    vah: float  # Value Area High
    val: float  # Value Area Low
    value_area_volume_pct: float  # Value Area 거래량 비율
    hvn_levels: List[float]  # High Volume Nodes
    lvn_levels: List[float]  # Low Volume Nodes
    profile: Dict[float, float]  # 가격대별 거래량
    total_volume: float


class VolumeProfile:
    """
    Volume Profile 분석기
    
    가격대별 거래량 분포를 계산하여 시장 구조 파악
    
    Args:
        tick_size: 가격 간격 (기본: 1.0)
        value_area_pct: Value Area 비율 (기본: 70%)
        hvn_threshold: HVN 판정 임계값 (평균 대비 배수)
        lvn_threshold: LVN 판정 임계값 (평균 대비 배수)
    """
    
    def __init__(
        self,
        tick_size: float = 1.0,
        value_area_pct: float = 0.70,
        hvn_threshold: float = 1.5,
        lvn_threshold: float = 0.5
    ):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
    
    def calculate(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> VolumeProfileResult:
        """
        Volume Profile 계산
        
        Args:
            prices: 가격 배열
            volumes: 거래량 배열
            
        Returns:
            VolumeProfileResult
        """
        # 가격대별 거래량 집계
        price_levels = self._discretize_prices(prices)
        profile = self._build_profile(price_levels, volumes)
        
        # POC (최대 거래량 가격대)
        poc = max(profile, key=profile.get)
        
        # Value Area 계산
        val, vah, value_area_volume = self._calculate_value_area(profile)
        
        total_volume = sum(profile.values())
        value_area_volume_pct = value_area_volume / total_volume if total_volume > 0 else 0
        
        # HVN / LVN 레벨
        hvn_levels, lvn_levels = self._find_hvn_lvn(profile)
        
        return VolumeProfileResult(
            poc=poc,
            vah=vah,
            val=val,
            value_area_volume_pct=value_area_volume_pct,
            hvn_levels=hvn_levels,
            lvn_levels=lvn_levels,
            profile=profile,
            total_volume=total_volume
        )
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ) -> VolumeProfileResult:
        """
        DataFrame에서 Volume Profile 계산
        
        Args:
            df: 가격/거래량 데이터프레임
            price_col: 가격 컬럼명
            volume_col: 거래량 컬럼명
            
        Returns:
            VolumeProfileResult
        """
        return self.calculate(
            df[price_col].values,
            df[volume_col].values
        )
    
    def _discretize_prices(self, prices: np.ndarray) -> np.ndarray:
        """가격을 tick_size 단위로 이산화"""
        return np.round(prices / self.tick_size) * self.tick_size
    
    def _build_profile(
        self,
        price_levels: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[float, float]:
        """가격대별 거래량 프로파일 구축"""
        profile = defaultdict(float)
        
        for price, volume in zip(price_levels, volumes):
            profile[price] += volume
        
        return dict(profile)
    
    def _calculate_value_area(
        self,
        profile: Dict[float, float]
    ) -> Tuple[float, float, float]:
        """
        Value Area 계산 (거래량 상위 70% 구간)
        
        Returns:
            (VAL, VAH, value_area_volume)
        """
        if not profile:
            return 0.0, 0.0, 0.0
        
        # POC 찾기
        poc = max(profile, key=profile.get)
        
        # 가격 정렬
        sorted_prices = sorted(profile.keys())
        total_volume = sum(profile.values())
        target_volume = total_volume * self.value_area_pct
        
        # POC에서 시작하여 양옆으로 확장
        poc_idx = sorted_prices.index(poc)
        lower_idx = poc_idx
        upper_idx = poc_idx
        accumulated_volume = profile[poc]
        
        while accumulated_volume < target_volume:
            # 아래쪽 볼륨
            lower_vol = profile[sorted_prices[lower_idx - 1]] if lower_idx > 0 else 0
            # 위쪽 볼륨
            upper_vol = profile[sorted_prices[upper_idx + 1]] if upper_idx < len(sorted_prices) - 1 else 0
            
            if lower_vol == 0 and upper_vol == 0:
                break
            
            # 더 큰 볼륨 방향으로 확장
            if lower_vol >= upper_vol and lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += lower_vol
            elif upper_idx < len(sorted_prices) - 1:
                upper_idx += 1
                accumulated_volume += upper_vol
            else:
                break
        
        val = sorted_prices[lower_idx]
        vah = sorted_prices[upper_idx]
        
        return val, vah, accumulated_volume
    
    def _find_hvn_lvn(
        self,
        profile: Dict[float, float]
    ) -> Tuple[List[float], List[float]]:
        """
        High Volume Node (HVN) 및 Low Volume Node (LVN) 찾기
        
        Returns:
            (hvn_levels, lvn_levels)
        """
        if not profile:
            return [], []
        
        volumes = list(profile.values())
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        hvn_threshold = avg_volume * self.hvn_threshold
        lvn_threshold = avg_volume * self.lvn_threshold
        
        hvn_levels = [
            price for price, vol in profile.items()
            if vol >= hvn_threshold
        ]
        
        lvn_levels = [
            price for price, vol in profile.items()
            if vol <= lvn_threshold and vol > 0
        ]
        
        return sorted(hvn_levels), sorted(lvn_levels)


class SessionVolumeProfile:
    """
    Session별 Volume Profile
    
    여러 세션(예: 아시아/유럽/미국)별로 Volume Profile 분석
    """
    
    def __init__(
        self,
        tick_size: float = 1.0,
        session_hours: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        self.tick_size = tick_size
        self.vp_calculator = VolumeProfile(tick_size=tick_size)
        
        # 기본 세션 시간 (UTC 기준)
        self.session_hours = session_hours or {
            'asia': (0, 8),
            'europe': (8, 16),
            'us': (16, 24)
        }
    
    def calculate_session_profiles(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume',
        datetime_col: str = 'timestamp'
    ) -> Dict[str, VolumeProfileResult]:
        """
        세션별 Volume Profile 계산
        
        Args:
            df: OHLCV 데이터프레임 (datetime 인덱스 또는 컬럼)
            price_col: 가격 컬럼
            volume_col: 거래량 컬럼
            datetime_col: 날짜시간 컬럼 (인덱스가 아닌 경우)
            
        Returns:
            {session_name: VolumeProfileResult}
        """
        # datetime 컬럼 생성
        if datetime_col in df.columns:
            df = df.copy()
            df['hour'] = pd.to_datetime(df[datetime_col]).dt.hour
        else:
            df = df.copy()
            df['hour'] = df.index.hour
        
        results = {}
        
        for session_name, (start_hour, end_hour) in self.session_hours.items():
            # 세션 데이터 필터링
            session_mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour)
            session_df = df[session_mask]
            
            if len(session_df) > 0:
                results[session_name] = self.vp_calculator.calculate(
                    session_df[price_col].values,
                    session_df[volume_col].values
                )
            else:
                # 빈 결과
                results[session_name] = VolumeProfileResult(
                    poc=0, vah=0, val=0,
                    value_area_volume_pct=0,
                    hvn_levels=[], lvn_levels=[],
                    profile={}, total_volume=0
                )
        
        return results


@numba.jit(nopython=True)
def calculate_tpo_profile(
    prices: np.ndarray,
    time_periods: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TPO (Time Price Opportunity) Profile 계산
    
    시간대별 가격 분포를 계산 (Numba 최적화)
    
    Args:
        prices: 가격 배열
        time_periods: 시간 구간 수
        
    Returns:
        (price_levels, tpo_counts)
    """
    # 가격 범위 계산
    min_price = np.min(prices)
    max_price = np.max(prices)
    
    # 가격 레벨 수
    num_levels = 100
    price_levels = np.linspace(min_price, max_price, num_levels)
    
    # TPO 카운트
    tpo_counts = np.zeros(num_levels, dtype=np.int32)
    
    # 각 시간 구간별 처리
    samples_per_period = len(prices) // time_periods
    
    for period in range(time_periods):
        start_idx = period * samples_per_period
        end_idx = start_idx + samples_per_period if period < time_periods - 1 else len(prices)
        
        period_prices = prices[start_idx:end_idx]
        
        # 이 기간의 고유 가격 레벨
        for price in period_prices:
            # 가장 가까운 레벨 찾기
            level_idx = np.argmin(np.abs(price_levels - price))
            tpo_counts[level_idx] += 1
    
    return price_levels, tpo_counts


def analyze_volume_profile_signals(
    current_price: float,
    vp_result: VolumeProfileResult
) -> Dict[str, any]:
    """
    Volume Profile 기반 트레이딩 시그널 생성
    
    Args:
        current_price: 현재 가격
        vp_result: Volume Profile 결과
        
    Returns:
        {
            'signal': 'bullish' / 'bearish' / 'neutral',
            'strength': 0-1,
            'reason': 시그널 이유,
            'key_levels': 주요 레벨
        }
    """
    signal = 'neutral'
    strength = 0.5
    reasons = []
    
    # POC 대비 가격 위치
    if current_price > vp_result.poc:
        if current_price > vp_result.vah:
            signal = 'bullish'
            strength = 0.7
            reasons.append(f"Price above VAH ({vp_result.vah:.2f})")
        else:
            signal = 'bullish'
            strength = 0.6
            reasons.append(f"Price above POC ({vp_result.poc:.2f})")
    elif current_price < vp_result.poc:
        if current_price < vp_result.val:
            signal = 'bearish'
            strength = 0.7
            reasons.append(f"Price below VAL ({vp_result.val:.2f})")
        else:
            signal = 'bearish'
            strength = 0.6
            reasons.append(f"Price below POC ({vp_result.poc:.2f})")
    
    # LVN 근처 (빠른 가격 이동 예상)
    for lvn in vp_result.lvn_levels:
        if abs(current_price - lvn) / current_price < 0.002:  # 0.2% 이내
            strength = min(strength + 0.1, 1.0)
            reasons.append(f"Near LVN ({lvn:.2f}) - Fast move expected")
            break
    
    # HVN 근처 (강한 지지/저항)
    for hvn in vp_result.hvn_levels:
        if abs(current_price - hvn) / current_price < 0.002:
            reasons.append(f"Near HVN ({hvn:.2f}) - Strong S/R")
            break
    
    key_levels = {
        'poc': vp_result.poc,
        'vah': vp_result.vah,
        'val': vp_result.val,
        'hvn': vp_result.hvn_levels[:3],  # 상위 3개
        'lvn': vp_result.lvn_levels[:3]
    }
    
    return {
        'signal': signal,
        'strength': strength,
        'reasons': reasons,
        'key_levels': key_levels
    }


if __name__ == "__main__":
    print("🧪 Testing Volume Profile...")
    
    # 샘플 데이터 생성 (정규분포 + 노이즈)
    np.random.seed(42)
    
    # 메인 트렌드 (50000 중심)
    main_prices = np.random.normal(50000, 100, 1000)
    main_volumes = np.random.uniform(1, 10, 1000)
    
    # 고거래량 구간 (49800, 50200)
    hvn_prices_1 = np.random.normal(49800, 20, 500)
    hvn_volumes_1 = np.random.uniform(10, 20, 500)
    
    hvn_prices_2 = np.random.normal(50200, 20, 500)
    hvn_volumes_2 = np.random.uniform(10, 20, 500)
    
    # 저거래량 구간 (50100)
    lvn_prices = np.random.normal(50100, 10, 100)
    lvn_volumes = np.random.uniform(0.1, 1, 100)
    
    # 전체 데이터 합치기
    all_prices = np.concatenate([main_prices, hvn_prices_1, hvn_prices_2, lvn_prices])
    all_volumes = np.concatenate([main_volumes, hvn_volumes_1, hvn_volumes_2, lvn_volumes])
    
    # Volume Profile 계산
    vp = VolumeProfile(tick_size=10.0, value_area_pct=0.70)
    result = vp.calculate(all_prices, all_volumes)
    
    print(f"✅ Volume Profile Results:")
    print(f"   - POC: {result.poc:.2f}")
    print(f"   - VAH: {result.vah:.2f}")
    print(f"   - VAL: {result.val:.2f}")
    print(f"   - Value Area Volume %: {result.value_area_volume_pct*100:.1f}%")
    print(f"   - Total Volume: {result.total_volume:.2f}")
    print(f"   - HVN Levels: {[f'{x:.2f}' for x in result.hvn_levels[:5]]}")
    print(f"   - LVN Levels: {[f'{x:.2f}' for x in result.lvn_levels[:5]]}")
    
    # 트레이딩 시그널 분석
    test_prices = [49700, 50000, 50300]
    
    print(f"\n✅ Trading Signals:")
    for price in test_prices:
        signals = analyze_volume_profile_signals(price, result)
        print(f"\n   Price: {price:.2f}")
        print(f"   - Signal: {signals['signal']} (strength: {signals['strength']:.2f})")
        print(f"   - Reasons: {', '.join(signals['reasons'])}")
    
    # TPO Profile 테스트
    print(f"\n✅ Testing TPO Profile...")
    price_levels, tpo_counts = calculate_tpo_profile(all_prices, time_periods=24)
    print(f"   - TPO Levels: {len(price_levels)}")
    print(f"   - Max TPO Count: {np.max(tpo_counts)}")
    print(f"   - TPO POC: {price_levels[np.argmax(tpo_counts)]:.2f}")
    
    print("\n🎉 Volume Profile test completed!")
