├──  README.md             # 프로젝트 설명 문서
│
├──  model/                # 모델 아키텍처 정의 및 실험
│   ├──  derainhaze.py     # Derain/Dehaze 모델 아키텍처
│   ├──  lightmodel.py     # 경량화(Lightweight) 모델 아키텍처
│   ├──  aodNet/           # AOD-Net 실험용 노트북
│   ├──  DBaodNet/         # DB-AOD-Net 실험용 노트북
│   └──  TBaodNet/         # TB-AOD-Net 버전별 실험 노트북 (v1~v6)
│
├──  preprocessing/        # 데이터 전처리 도구
│   ├──  prepros.py        # 전처리 메인 로직
│   ├──  resize.py         # 이미지 리사이징 (v1)
│   ├──  resize2.py        # 이미지 리사이징 (v2)
│   ├──  split_dataset.py  # 데이터셋 분할 (Train/Val)
│   └──  resized_image_list.txt # 리사이징 로그
│
├──  train/                # 모델 학습 스크립트
│   ├──  detrain.py        # 메인 모델 학습 실행
│   ├──  detrainLite.py    # 경량화 모델 학습 실행 (v1)
│   └──  detrainLite2.py   # 경량화 모델 학습 실행 (v2)
│
└──  utils/                # 보조 기능 및 유틸리티 모음
    ├──  datatest.py       # 데이터 테스트 스크립트 (v1)
    ├──  datatest2.py      # 데이터 테스트 스크립트 (v2)
    ├──  downloadYoloData.py # 데이터 다운로드 도구
    ├──  points.py         # 포인트 처리 로직 (Main)
    ├──  pointsLite.py     # 포인트 처리 로직 (Lite)
    └──  testimage.py      # 이미지 테스트 및 추론 도구