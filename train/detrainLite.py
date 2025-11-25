import os
import sys
from glob import glob
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from torchvision.models import VGG16_Weights

# 프로젝트 루트 경로를 sys.path에 추가 (train 폴더에서 상위 모듈 참조용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [수정] 폴더 구조에 맞게 import 경로 변경
import model.lightmodel as lightmodel
import preprocessing.prepros as prepros

# import VGGPerceptual

# ================================================
# 3) 학습 예시: DataLoader + 학습 루프 (Windows 멀티프로세싱 안전 진입점)
# ================================================
if __name__ == "__main__":
    # 1) 사용자 설정: 경로 및 해상도
    root_gt_folder = "./dataset_split/train/gt"
    root_rain_folder = "./dataset_split/train/input"
    val_gt_folder   = "./dataset_split/test/gt"
    val_rain_folder = "./dataset_split/test/input"
    
    # [수정] 모델 저장 경로 설정 (Lite 모델용 이름으로 저장)
    save_dir = "pt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "Litemodel.pt")

    # Lite 모델용 해상도 (유지함)
    img_h, img_w = 320, 480
    batch_size       = 8
    val_batch_size   = 4
    num_epochs       = 50
    lr               = 1e-3
    
    print("===== Lite 모델 학습 스크립트 시작 =====")
    print(f"[Config] GT 폴더: {root_gt_folder}")
    print(f"[Config] Rain 폴더: {root_rain_folder}")
    print(f"[Config] 이미지 크기: ({img_h}, {img_w})\n")

    # 2) 데이터셋 및 DataLoader 생성
    print("[Main] 데이터셋 생성 중...")
    
    # --- ⬇️ 데이터셋 분할 로직 (주석에 맞게 10% 사용으로 수정) ⬇️ ---

    # 2-1) 전체 훈련 데이터셋 불러오기
    full_dataset = prepros.RainDSSynDataset(
        root_gt=root_gt_folder,
        root_rain=root_rain_folder,
        img_size=(img_h, img_w),
        transform=None  # 기본 Resize + ToTensor 사용
    )

    # 2-2) 데이터셋을 훈련용(10%)과 나머지(90%)로 분할
    dataset_size = len(full_dataset)
    subset_size = int(dataset_size * 0.1)  # 10% 사용
    if subset_size == 0 and dataset_size > 0: subset_size = 1
    remaining_size = dataset_size - subset_size

    # 2-3) random_split을 사용하여 분할
    train_subset, _ = random_split(full_dataset, [subset_size, remaining_size])
    
    print(f"[Main] 전체 데이터셋 크기: {dataset_size}")
    print(f"[Main] 실습용 훈련 데이터 크기(10%): {len(train_subset)}")
    print("[Main] DataLoader 설정 중...")
    
    # 2-4) DataLoader에 train_subset 전달
    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,    # Windows 호환성을 위해 0 권장
        pin_memory=True
    )
    
    # --- ⬆️ 수정 완료 ⬆️ ---

    print(f"[Main] 데이터로더 크기: {len(loader)} 배치\n")
    
    val_dataset = prepros.RainDSSynDataset(
        root_gt=val_gt_folder,
        root_rain=val_rain_folder,
        img_size=(img_h, img_w),
        transform=transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 3) 장치(device) 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}\n")

    # 4) 모델, 옵티마이저, 손실함수 정의
    print("[Main] 모델 초기화 중 (Lite)...")
    model = lightmodel.DerainNetLite().to(device)
    print("[Main] 옵티마이저 및 손실 함수 설정 중...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    print("[Main] 준비 완료\n")

    # 5) 에폭 수 정의
    print(f"[Main] 학습 에폭 수: {num_epochs}\n")
    
    # λ (Perceptual Loss 가중치) 설정
    lambda_perc = 0.01
    
    # 6) 학습 루프
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] 학습 시작")
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            rain_imgs = batch["rain"].to(device)  # (B,3,H,W)
            gt_imgs   = batch["gt"].to(device)    # (B,3,H,W)

            optimizer.zero_grad()
            outputs = model(rain_imgs)            # (B,3,H,W)
            
            # 1) 픽셀 MSE 손실
            loss_mse = criterion(outputs, gt_imgs)
            
            loss = loss_mse # 일단 MSE만 사용
            
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * rain_imgs.size(0)

            # 배치 단위 진행 상황 출력
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"   [Batch {batch_idx}/{len(loader)}]   Loss: {batch_loss:.6f}")

        # 평균 Loss 계산
        epoch_loss /= len(train_subset)
        print(f"[Epoch {epoch+1}/{num_epochs}]   평균 Loss: {epoch_loss:.6f}\n")
    
    # 7) 학습 완료 후 TorchScript로 저장 (추론용)
    print("[Main] 학습 완료, TorchScript 모델로 변환 중...")
    model.eval()
    example = torch.randn(1, 3, img_h, img_w).to(device)
    
    # CPU로 이동하여 저장 (호환성 확보)
    model_cpu = model.cpu()
    example_cpu = example.cpu()
    
    traced = torch.jit.trace(model_cpu, example_cpu)
    traced.save(save_path)
    print(f"[Main] TorchScript 모델 저장 완료: {save_path}")
    print("===== Lite 모델 학습 스크립트 종료 =====")