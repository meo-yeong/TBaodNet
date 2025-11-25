import os
from glob import glob
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split # 👈 수정: random_split 추가
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from torchvision.models import VGG16_Weights
import model.derainhaze as derainhaze
import prepros
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
    img_h, img_w = 480, 720
    batch_size       = 8
    val_batch_size   = 4
    num_epochs       = 50
    lr               = 1e-3
    print("===== 학습 스크립트 시작 =====")
    print(f"[Config] GT 폴더: {root_gt_folder}")
    print(f"[Config] Rain 폴더: {root_rain_folder}")
    print(f"[Config] 이미지 크기: ({img_h}, {img_w})\n")

    # 2) 데이터셋 및 DataLoader 생성
    print("[Main] 데이터셋 생성 중...")
    
    # --- ⬇️ 여기부터 수정된 부분입니다 ⬇️ ---

    # 2-1) 먼저 전체 훈련 데이터셋을 불러옵니다.
    full_dataset = prepros.RainDSSynDataset(
        root_gt=root_gt_folder,
        root_rain=root_rain_folder,
        img_size=(img_h, img_w),
        transform=None  # 기본 Resize + ToTensor 사용
    )

    # 2-2) 데이터셋을 훈련용(10%)과 나머지(90%)로 나눌 크기를 계산합니다.
    dataset_size = len(full_dataset)
    subset_size = dataset_size // 10
    remaining_size = dataset_size - subset_size

    # 2-3) random_split을 사용하여 무작위로 데이터셋을 분할합니다.
    #      _ 변수를 사용해 90%에 해당하는 데이터는 무시합니다.
    train_subset, _ = random_split(full_dataset, [subset_size, remaining_size])
    
    print(f"[Main] 전체 데이터셋 크기: {dataset_size}, 사용할 훈련 데이터 크기: {len(train_subset)}")
    print("[Main] DataLoader 설정 중...")
    
    # 2-4) DataLoader에 전체 데이터셋 대신 10%만 분할한 train_subset을 전달합니다.
    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,    # 워커 수 (Windows에서는 0 또는 1로 줄여서 테스트 권장)
        pin_memory=True
    )
    
    # --- ⬆️ 여기까지가 수정된 부분입니다 ⬆️ ---

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
    print("[Main] 모델 초기화 중...")
    model = derainhaze.DerainNet().to(device)
    print("[Main] 옵티마이저 및 손실 함수 설정 중...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    print("[Main] 준비 완료\n")

    # 5) 에폭 수 정의
    print(f"[Main] 학습 에폭 수: {num_epochs}\n")
    # Perceptual Loss용 VGG 추출기 생성
    # vgg_extractor = VGGPerceptual(requires_grad=False).to(device)

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
            # 2) Perceptual Loss
            # loss_perc = VGGPerceptual.perceptual_loss(outputs, gt_imgs, vgg_extractor)
            # 3) 총 손실 = MSE + λ * Perceptual
            loss = loss_mse + lambda_perc # * loss_perc

            #loss = criterion(outputs, gt_imgs)   # 단순 MSE 손실
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            # ⚠️ 수정: epoch_loss를 계산할 때 len(dataset)이 아닌 len(train_subset)을 사용해야 정확합니다.
            epoch_loss += batch_loss * rain_imgs.size(0)

            # 배치 단위 진행 상황 출력
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"   [Batch {batch_idx}/{len(loader)}]   Loss: {batch_loss:.6f}")

        # ⚠️ 수정: 평균 Loss를 계산할 때 len(dataset) 대신 len(train_subset)을 사용합니다.
        epoch_loss /= len(train_subset)
        print(f"[Epoch {epoch+1}/{num_epochs}]   평균 Loss: {epoch_loss:.6f}\n")
    
    # 7) 학습 완료 후 TorchScript로 저장 (추론용)
    print("[Main] 학습 완료, TorchScript 모델로 변환 중...")
    model.eval()
    example = torch.randn(1, 3, img_h, img_w).to("cpu")
    traced = torch.jit.trace(model.cpu(), example)
    traced.save("dedrop_derain_dehaze.pt")
    print("[Main] TorchScript 모델 저장 완료: dedrop_derain_dehaze.pt")
    print("===== 학습 스크립트 종료 =====")