import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가 (utils 폴더에서 상위 폴더의 모듈을 import 하기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.derainhaze

def test_single_image(model_weights_path, input_image_path, output_image_path):
    """
    단일 이미지에 대해 비 제거 추론을 수행하고 결과를 저장합니다.
    """
    # 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    # 모델 초기화 및 디바이스로 이동
    model = model.derainhaze.DerainNet().to(device)

    # 학습된 가중치 불러오기
    try:
        # 주의: 저장된 파일이 state_dict 형태라고 가정합니다.
        checkpoint = torch.load(model_weights_path, map_location=device)
        
        # 체크포인트가 dict 형태(키값 포함)일 경우 처리
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"'{model_weights_path}'에서 모델 가중치를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{model_weights_path}' 경로에 모델 가중치 파일이 없습니다.")
        return
    except Exception as e:
        print(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    # 모델을 평가 모드로 설정
    model.eval()

    # 이미지 전처리를 위한 변환 정의
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 이미지 불러오기 및 전처리
    try:
        input_image = Image.open(input_image_path).convert('RGB')
        print(f"'{input_image_path}' 이미지를 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{input_image_path}' 경로에 입력 이미지가 없습니다.")
        return

    input_tensor = transform(input_image)
    # 모델은 (배치, 채널, 높이, 너비) 형태의 입력을 기대하므로 배치 차원 추가
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 추론 실행 (gradient 계산 비활성화)
    with torch.no_grad():
        output_tensor = model(input_batch)

    # 결과 후처리
    # 배치 차원 제거
    output_tensor = output_tensor.squeeze(0)
    # 픽셀 값이 [0, 1] 범위를 벗어나지 않도록 클램핑
    output_tensor = torch.clamp(output_tensor, 0, 1)
    # Tensor를 다시 PIL 이미지로 변환
    output_image = transforms.ToPILImage()(output_tensor.cpu())

    # 결과 이미지 저장
    # 출력 경로의 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    output_image.save(output_image_path)
    print(f"비가 제거된 이미지를 '{output_image_path}'에 저장했습니다. 🎉")


if __name__ == '__main__':
    
    # 1. 학습된 모델 가중치 파일 경로 (프로젝트 구조에 맞게 수정됨)
    # 예: pt 폴더 내의 모델 파일
    MODEL_WEIGHTS_PATH = os.path.join('pt', 'dedrop_derain_dehaze.pt')

    # 2. 비를 제거할 입력 이미지 경로 (테스트용 이미지)
    # 예: dataset_split/val/input/8_rain.png (points.py와 동일하게 맞춤)
    INPUT_IMAGE_PATH = os.path.join('dataset_split', 'val', 'input', '8_rain.png')

    # 3. 결과 이미지를 저장할 경로
    OUTPUT_IMAGE_PATH = os.path.join('processedImg', 'derained_single_result.png')
    
    # 함수 실행
    # (파일이 실제로 존재하는지 확인 후 실행하는 것이 좋습니다)
    if os.path.exists(INPUT_IMAGE_PATH) and os.path.exists(MODEL_WEIGHTS_PATH):
        test_single_image(MODEL_WEIGHTS_PATH, INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    else:
        print("🚨 경로 확인 필요:")
        if not os.path.exists(MODEL_WEIGHTS_PATH): print(f" - 모델 없음: {MODEL_WEIGHTS_PATH}")
        if not os.path.exists(INPUT_IMAGE_PATH): print(f" - 입력 이미지 없음: {INPUT_IMAGE_PATH}")