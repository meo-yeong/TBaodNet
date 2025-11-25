import torch
import cv2
import numpy as np
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가 (utils 폴더에서 상위 폴더의 모듈을 import 하기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import model.lightmodel as lightmodel

def load_trained_model(path, device):
    """
    - 파일 확장자가 '.pt'이면 torch.jit.load()를 시도
    - 그렇지 않으면 torch.load()로 state_dict를 불러와 직접 로드
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")

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
    print("===== 경량화(Lite) 모델 추론 스크립트 시작 =====")

    # (1) device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Using device: {device}")

    # (2) 경로 설정 (★★★ 사용 전 실제 파일 경로로 수정해주세요 ★★★)
    # -------------------------------------------------------------------
    # 학습된 Lite 모델 경로
    trained_path = os.path.join("pt", "Litemodel.pt")
    
    # 테스트할 비 오는 이미지 경로
    sample_rain_img = os.path.join("dataset_split", "val", "input", "8_rain.png")
    
    # 정답(Clean) 이미지 경로
    gt_clean_img = os.path.join("dataset_split", "val", "gt", "8_rain.png")
    
    # 결과 이미지를 저장할 경로
    output_dir = "processedImg"
    output_filename = "processed_image2.jpg"
    output_path = os.path.join(output_dir, output_filename)
    # -------------------------------------------------------------------

    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[Info] '{output_dir}' 폴더가 없어 새로 생성했습니다.")

    # 모델 로드
    try:
        model = load_trained_model(trained_path, device)
    except Exception as e:
        print(f"[Error] 모델 로드 실패: {e}")
        sys.exit(1)

    # (3) 추론할 이미지 및 정답 이미지 불러오기
    print(f"[Inference] 처리할 이미지: {sample_rain_img}")
    print(f"[Evaluation] 정답 이미지: {gt_clean_img}")

    # (4-1) 입력 이미지(비 오는) 열기 및 전처리
    img_bgr = cv2.imread(sample_rain_img)
    if img_bgr is None:
        print(f"[Error] 입력 이미지를 찾을 수 없습니다: {sample_rain_img}")
        sys.exit(1)
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = 480, 720  # 모델 입력 크기 (height, width)
    img_resized = cv2.resize(img_rgb, (W, H))
    img_f = img_resized.astype(np.float32) / 255.0
    
    # (4-2) 정답 이미지(깨끗한) 열기 및 전처리
    gt_bgr = cv2.imread(gt_clean_img)
    if gt_bgr is None:
        print(f"[Warning] 정답 이미지를 찾을 수 없습니다. PSNR/SSIM 계산을 건너뜁니다: {gt_clean_img}")
        gt_f = None
    else:
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        gt_resized = cv2.resize(gt_rgb, (W, H))
        gt_f = gt_resized.astype(np.float32) / 255.0

    # 모델 추론
    input_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 출력 텐서를 이미지 형태로 변환
    output_img_f = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_img_f = np.clip(output_img_f, 0.0, 1.0)

    # --- SSIM 및 PSNR 계산 ---
    if gt_f is not None:
        gt_img_for_metrics = gt_f
        processed_img_for_metrics = output_img_f

        # PSNR 계산
        psnr_value = calculate_psnr(gt_img_for_metrics, processed_img_for_metrics, data_range=1.0)
        print(f"계산된 PSNR: {psnr_value:.4f}")

        # SSIM 계산
        ssim_value = calculate_ssim(gt_img_for_metrics, processed_img_for_metrics, data_range=1.0, channel_axis=2)
        print(f"계산된 SSIM: {ssim_value:.4f}")
    else:
        print("정답 이미지가 없어 PSNR/SSIM을 계산하지 않았습니다.")

    # --- 결과 이미지 저장 ---
    output_img_uint8 = (output_img_f * 255).astype(np.uint8)
    output_img_bgr = cv2.cvtColor(output_img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, output_img_bgr)
    print(f"처리된 이미지 저장 완료: {output_path}")

    print("===== 추론 스크립트 종료 =====")