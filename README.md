# CoT기반 특허가치 예측


## 도커 이미지 내 파일 구성
```bash
.
├── data
├── output
├── .python-version
├── model_merge_adapter.py
├── model_test.ipynb
├── model_train.py
├── pyproject.toml
└── README.md
```

1. `data`: 원천 데이터 및 학습/검증/테스트 데이터셋
    - `가공데이터`: 원천 데이터
    - Docker container에 volume으로 연결됩니다.
2. `output`: 학습된 모델
    - Docker container에 volume으로 연결됩니다.
3. `.python-version`, `pyproject.toml`: 의존성 관리 파일들
4. `model_test.ipynb`: 모델 검증 실행파일
5. `model_train.py`: 학습 실행파일
6. `model_merge_adapter.py`: 학습된 LoRA 어댑터와 기본 모델을 합쳐 최종 모델을 생성하는 실행파일


## 유효성 검증 모델 학습 및 검증 조건
- 개발 언어: Python 3.12
- 프레임워크: CUDA 12.8
- 학습 알고리즘: gemma-3-1b-it
- 학습조건
    1. epochs: 1
    2. per_device_train_batch_size: 2
    3. gradient_accumulation_steps: 8
    4. optim: "adamw_8bit"
    5. warmup_steps: 30
    6. learning_rate: 2e-4
    7. bf16: True


## 가상환경 활성화
```bash
cd /app
uv sync
source .venv/bin/activate
```
