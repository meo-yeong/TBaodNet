import torch
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import model.lightmodel as lightmodel

def load_trained_model(path, device):
    """
    - 파일 확장자가 '.pt'이면 torch.jit.load()를 시도
    - 그렇지 않으면 torch.load()로 state_dict를 불러와 직접 로드
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pt":
        print(f"[Load] TorchScript 아카이브 '{path}' 로드 중...")
        model = torch.jit.load(path, map_location=device)
        model.to(device)
        model.eval()
        print("[Load] TorchScript 모델 로드 완료 (eval 모드).\n")
        return model

    else:
        print(f"[Load] state_dict 아카이브 '{path}' 로드 중...")
        model = lightmodel.DerainNetLite().to(device)
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        print("[Load] state_dict 모델 로드 완료 (eval 모드).\n")
        return model

if __name__ == "__main__":
    print("===== 추론 스크립트 시작 =====")

    # (1) device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Using device: {device}")

    # (2) 경로 설정
    trained_path = "./pt/Litemodel.pt"
    sample_rain_img = "./dataset_split/val/input/8_rain.png"
    gt_clean_img = "./dataset_split/val/gt/8_rain.png"  # 👈 [수정] 정답(Ground Truth) 이미지 경로 추가
    output_path = "./processedImg/processed_image2.jpg"
    
    # 모델 로드
    model = load_trained_model(trained_path, device)

    # (3) 추론할 이미지 및 정답 이미지 불러오기
    print(f"[Inference] 처리할 이미지: {sample_rain_img}")
    print(f"[Evaluation] 정답 이미지: {gt_clean_img}") # 👈 [추가]

    # (4-1) 입력 이미지(비 오는) 열기 및 전처리
    img_bgr = cv2.imread(sample_rain_img)
    if img_bgr is None:
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {sample_rain_img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = 480, 720  # 모델 입력 크기 (height, width)
    img_resized = cv2.resize(img_rgb, (W, H))
    img_f = img_resized.astype(np.float32) / 255.0
    
    # 👈 [추가] (4-2) 정답 이미지(깨끗한) 열기 및 전처리
    gt_bgr = cv2.imread(gt_clean_img)
    if gt_bgr is None:
        raise FileNotFoundError(f"정답 이미지를 찾을 수 없습니다: {gt_clean_img}")
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    gt_resized = cv2.resize(gt_rgb, (W, H)) # 입력과 동일한 크기로 리사이즈
    gt_f = gt_resized.astype(np.float32) / 255.0

    # 모델 추론
    input_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 출력 텐서를 이미지 형태로 변환
    output_img_f = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img_f = np.clip(output_img_f, 0.0, 1.0)

    # --- SSIM 및 PSNR 계산 코드 (수정됨) ---

    # 정답 이미지 (0~1 범위 float32)
    gt_img_for_metrics = gt_f  # 👈 [수정] 'original_img'가 아닌 'gt_img' 사용

    # 처리된 이미지 (0~1 범위 float32)
    processed_img_for_metrics = output_img_f

    # PSNR 계산 (정답 이미지와 모델 출력 비교)
    psnr_value = calculate_psnr(gt_img_for_metrics, processed_img_for_metrics, data_range=1.0)
    print(f"계산된 PSNR: {psnr_value:.4f}")

    # SSIM 계산 (정답 이미지와 모델 출력 비교)
    ssim_value = calculate_ssim(gt_img_for_metrics, processed_img_for_metrics, data_range=1.0, channel_axis=2)
    print(f"계산된 SSIM: {ssim_value:.4f}")

    # --- 결과 이미지 저장 ---
    output_img_uint8 = (output_img_f * 255).astype(np.uint8)
    output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, output_img_bgr)
    print(f"처리된 이미지 저장 완료: {output_path}")

    print("===== 추론 스크립트 종료 =====")