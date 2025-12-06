"""
프로덕션 AI 모델 학습 스크립트 (최고 성능)

3개 모델을 순차적으로 학습:
1. Oracle (TFT) - 가격 예측
2. Strategist (Decision Transformer) - 행동 최적화  
3. Guardian (Contrastive VAE) - 시장 체제 감지

사용법:
    # 전체 학습
    python scripts/train_production_models.py --all

    # 개별 학습
    python scripts/train_production_models.py --model oracle
    python scripts/train_production_models.py --model strategist
    python scripts/train_production_models.py --model guardian
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print('='*70)
print('🚀 PROJECT QUANTUM ALPHA - AI MODEL TRAINING')
print('='*70)


def train_oracle():
    """Oracle (TFT) 학습 - 가격 예측"""
    print('\n' + '='*70)
    print('1️⃣  ORACLE (TFT) TRAINING - Price Prediction')
    print('='*70)
    
    from ai.training.oracle_trainer import OracleTrainer
    
    config = {
        # 데이터
        'data_dir': 'data/historical_5min_features',
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'train_years': [2019, 2020, 2021, 2022, 2023],
        'test_year': 2024,
        
        # 모델 아키텍처 (최고 성능)
        'encoder_length': 60,        # 5시간 히스토리
        'decoder_length': 24,        # 2시간 예측
        'hidden_size': 128,          # 큰 hidden dimension
        'attention_heads': 4,        # Multi-head attention
        'num_layers': 3,             # 깊은 네트워크
        'dropout': 0.1,
        'hidden_continuous_size': 64,
        
        # 학습 설정 (최고 성능)
        'batch_size': 256,           # 큰 배치
        'learning_rate': 0.001,
        'max_epochs': 100,
        'early_stopping_patience': 15,
        'gradient_clip_val': 0.1,
        
        # GPU 설정
        'use_gpu': True,
        'precision': 16,             # Mixed precision
        'num_workers': 8,            # 데이터 로딩 병렬화
        
        # 출력
        'output_dir': 'models/oracle',
        'save_top_k': 3,             # 상위 3개 모델 저장
        'log_every_n_steps': 50,
    }
    
    trainer = OracleTrainer(config)
    trainer.train()
    
    print('\n✅ Oracle training completed!')
    print(f'   Model saved to: {config["output_dir"]}/best_model.ckpt')


def train_strategist():
    """Strategist (Decision Transformer) 학습 - 행동 최적화"""
    print('\n' + '='*70)
    print('2️⃣  STRATEGIST (Decision Transformer) TRAINING - Action Optimization')
    print('='*70)
    
    from ai.training.strategist_trainer import StrategistTrainer
    
    config = {
        # 데이터
        'data_dir': 'data/historical_5min_features',
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'train_years': [2019, 2020, 2021, 2022, 2023],
        'test_year': 2024,
        
        # 모델 아키텍처 (최고 성능)
        'context_length': 90,        # 7.5시간 컨텍스트
        'hidden_size': 256,          # 큰 representation
        'num_layers': 6,             # 깊은 Transformer
        'num_heads': 8,              # Multi-head attention
        'dropout': 0.1,
        
        # RL 설정
        'discount_factor': 0.99,
        'reward_scale': 1.0,
        'rtg_scale': 1000.0,         # Return-to-go 스케일
        
        # 학습 설정 (최고 성능)
        'batch_size': 128,
        'learning_rate': 0.0001,     # 낮은 학습률 (안정성)
        'max_epochs': 200,           # RL은 오래 필요
        'early_stopping_patience': 20,
        'gradient_clip_val': 1.0,
        
        # GPU 설정
        'use_gpu': True,
        'precision': 16,
        'num_workers': 8,
        
        # 출력
        'output_dir': 'models/strategist',
        'save_top_k': 3,
        'log_every_n_steps': 100,
    }
    
    trainer = StrategistTrainer(config)
    trainer.train()
    
    print('\n✅ Strategist training completed!')
    print(f'   Model saved to: {config["output_dir"]}/best_model.ckpt')


def train_guardian():
    """Guardian (Contrastive VAE) 학습 - 시장 체제 감지"""
    print('\n' + '='*70)
    print('3️⃣  GUARDIAN (Contrastive VAE) TRAINING - Market Regime Detection')
    print('='*70)
    
    from ai.training.guardian_trainer import GuardianTrainer
    
    config = {
        # 데이터
        'data_dir': 'data/historical_5min_features',
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'train_years': [2019, 2020, 2021, 2022, 2023],
        'test_year': 2024,
        
        # 모델 아키텍처 (최고 성능)
        'latent_dim': 64,            # 잠재 공간 차원
        'hidden_dims': [256, 128, 64], # Encoder/Decoder
        'window_size': 120,          # 10시간 윈도우
        
        # VAE & Contrastive 설정
        'beta': 4.0,                 # VAE beta (KL weight)
        'temperature': 0.5,          # Contrastive learning 온도
        
        # 학습 설정 (최고 성능)
        'batch_size': 512,           # 매우 큰 배치
        'learning_rate': 0.001,
        'max_epochs': 100,
        'early_stopping_patience': 10,
        'gradient_clip_val': 0.5,
        
        # GPU 설정
        'use_gpu': True,
        'precision': 16,
        'num_workers': 8,
        
        # 출력
        'output_dir': 'models/guardian',
        'save_top_k': 3,
        'log_every_n_steps': 50,
    }
    
    trainer = GuardianTrainer(config)
    trainer.train()
    
    print('\n✅ Guardian training completed!')
    print(f'   Model saved to: {config["output_dir"]}/best_model.ckpt')


def main():
    parser = argparse.ArgumentParser(description='Train production AI models')
    parser.add_argument('--model', type=str, choices=['oracle', 'strategist', 'guardian', 'all'],
                       default='all', help='Model to train')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU (default: True)')
    
    args = parser.parse_args()
    
    # GPU 확인
    import torch
    if args.gpu and torch.cuda.is_available():
        print(f'\n🔥 GPU Available: {torch.cuda.get_device_name(0)}')
        print(f'   CUDA Version: {torch.version.cuda}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('\n⚠️  Running on CPU (will be slower)')
    
    # 학습 시작
    if args.model == 'all':
        print('\n📋 Training all 3 models sequentially...')
        print('   Estimated total time: 14-24 hours')
        
        train_guardian()    # 가장 빠름 (2-4시간)
        train_oracle()      # 중간 (4-8시간)
        train_strategist()  # 가장 오래 (8-12시간)
        
        print('\n' + '='*70)
        print('🎉 ALL MODELS TRAINING COMPLETED!')
        print('='*70)
        print('\n📁 Trained models:')
        print('   - models/oracle/best_model.ckpt')
        print('   - models/strategist/best_model.ckpt')
        print('   - models/guardian/best_model.ckpt')
        print('\n📋 Next steps:')
        print('   1. Evaluate models: python scripts/evaluate_models.py')
        print('   2. Run backtest: python scripts/backtest_ensemble.py')
        print('   3. Convert to ONNX: python scripts/convert_to_onnx.py')
        
    elif args.model == 'oracle':
        train_oracle()
    elif args.model == 'strategist':
        train_strategist()
    elif args.model == 'guardian':
        train_guardian()


if __name__ == '__main__':
    main()
