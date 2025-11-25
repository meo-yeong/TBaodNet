```text
.
├──  .gitignore            # Git 업로드 제외 설정
├──  README.md             # 프로젝트 설명 문서
│
├──  model/                # 모델 아키텍처 정의
│   ├──  derainhaze.py     # Main 모델(DerainNet) 클래스 정의
│   ├──  lightmodel.py     # Lite 경량화 모델(DerainNetLite) 클래스 정의
│   ├──  aodNet/           # AOD-Net 실험용 노트북
│   ├──  DBaodNet/         # DB-AOD-Net 실험용 노트북
│   └──  TBaodNet/         # TB-AOD-Net 버전별 실험 기록
│
├──  preprocessing/        # 데이터 전처리 및 로더
│   ├──  prepros.py        # Custom Dataset 클래스 (RainDSSynDataset)
│   ├──  resize.py         # 이미지 리사이즈 + 파일명 목록(txt) 생성
│   ├──  resize2.py        # 이미지 리사이즈 (tqdm 진행바 포함, txt 생성 안 함)
│   ├──  split_dataset.py  # txt 목록 기반 Train(7):Test(2):Val(1) 데이터 분할
│   └──  resized_image_list.txt # resize.py 실행 시 생성되는 파일 목록
│
├──  train/                # 모델 학습 스크립트
│   ├──  detrain.py        # Main 모델 학습 (데이터 10% 사용, dedrop_derain_dehaze.pt 저장)
│   ├──  detrainLite.py    # Lite 모델 학습 (데이터 10% 사용, Litemodel.pt 저장)
│   └──  detrainLite2.py   # Lite 모델 학습 (데이터 100% 사용, Litemodel2.pt 저장)
│
└──  utils/                # 유틸리티 및 평가 도구
    ├──  points.py         # Main 모델 평가 (PSNR/SSIM 측정, dedrop_derain_dehaze.pt 로드)
    ├──  pointsLite.py     # Lite 모델 평가 (PSNR/SSIM 측정, Litemodel.pt 로드)
    ├──  testimage.py      # 단일 이미지 추론 테스트
    ├──  datatest.py       # 데이터셋 준비 1: txt 목록에 있는 이미지를 input 폴더로 복사
    ├──  datatest2.py      # 데이터셋 준비 2: 복사된 이미지 파일명 정규화
    └──  downloadYoloData.py # 외부 데이터셋(YOLO/COCO) 다운로드 도구