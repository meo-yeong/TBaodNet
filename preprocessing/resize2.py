import cv2
import os
from tqdm import tqdm # 진행 상황 표시를 위해 tqdm 추가

def batch_resize_images(input_folder, output_folder, target_width, target_height):
    """
    폴더 내의 모든 이미지를 지정된 해상도로 리사이즈합니다.

    :param input_folder: 원본 이미지 폴더 경로
    :param output_folder: 리사이즈된 이미지를 저장할 폴더 경로
    :param target_width: 목표 너비 (width)
    :param target_height: 목표 높이 (height)
    """
    # 결과물을 저장할 폴더가 없으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 입력 폴더의 파일 목록을 가져옵니다.
    file_list = os.listdir(input_folder)
    print(f"총 {len(file_list)}개의 파일에 대해 리사이즈를 시작합니다.")

    # 입력 폴더 내의 모든 파일을 순회합니다.
    for filename in tqdm(file_list, desc="Resizing images"):
        # 지원하는 이미지 확장자인지 확인합니다.
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # 이미지 파일 경로를 조합합니다.
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # OpenCV로 이미지를 읽어옵니다.
                image = cv2.imread(input_path)
                if image is None:
                    print(f"'{filename}' 파일을 읽을 수 없습니다. 건너뜁니다.")
                    continue

                # 이미지 리사이즈 (축소 시에는 INTER_AREA가 권장됩니다)
                resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

                # 리사이즈된 이미지를 저장합니다.
                cv2.imwrite(output_path, resized_image)

            except Exception as e:
                print(f"'{filename}' 처리 중 오류 발생: {e}")
    
    print(f"\n모든 작업이 완료되었습니다. 결과는 '{output_folder}' 폴더에 저장되었습니다.")


# --- 사용 방법 ---
# 1. 원본 이미지가 있는 폴더 경로를 지정합니다.  test train val 순서
input_directory = "dataset_split/val/gt"

# 2. 리사이즈된 이미지를 저장할 폴더 경로를 지정합니다.
output_directory = "processdata/val/gt"

# 3. 목표 해상도를 지정합니다 (너비, 높이 순).
width = 360
height = 240

# 4. 함수를 실행합니다.
batch_resize_images(input_directory, output_directory, width, height)