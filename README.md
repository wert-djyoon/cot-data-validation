# CoT기반 특허가치 예측 프로젝트

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

1. `data`: 학습/검증/테스트 데이터셋
    - Docker container에 volume으로 연결됩니다.
2. `output`: 학습된 모델
    - Docker container에 volume으로 연결됩니다.
3. `.python-version`, `pyproject.toml`: 의존성 관리 파일들
4. `model_test.ipynb`: 모델 검증 실행파일
5. `model_train.py`: 학습 실행파일
6. `model_merge_adapter.py`: 학습된 LoRA 어댑터와 기본 모델을 합쳐 최종 모델을 생성하는 실행파일
