import cv2
import os

def batch_resize_and_log_images(input_folder, output_folder, target_width, target_height, log_file_path):
    """
    폴더 내의 모든 이미지를 지정된 해상도로 리사이즈하고,
    처리된 파일명을 텍스트 파일에 기록합니다.

    :param input_folder: 원본 이미지 폴더 경로
    :param output_folder: 리사이즈된 이미지를 저장할 폴더 경로
    :param target_width: 목표 너비 (width)
    :param target_height: 목표 높이 (height)
    :param log_file_path: 파일명 목록을 저장할 .txt 파일 경로
    """
    # 결과물을 저장할 폴더가 없으면 생성합니다.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 파일명을 기록할 텍스트 파일을 쓰기 모드('w')로 엽니다.
    # (주의: 'a' 모드는 기존 내용 뒤에 추가합니다. 새로 만드려면 'w'를 쓰세요)
    # 여기서는 기존 코드의 'a'를 유지합니다.
    with open(log_file_path, 'a') as log_file:
        # 입력 폴더 내의 모든 파일을 순회합니다.
        for filename in os.listdir(input_folder):
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
                    
                    # 성공적으로 처리된 파일명을 txt 파일에 기록합니다.
                    log_file.write(filename + '\n')
                    
                    print(f"'{filename}' 파일 리사이즈 및 로그 기록 완료 -> {output_path}")

                except Exception as e:
                    print(f"'{filename}' 처리 중 오류 발생: {e}")
    
    print(f"\n모든 작업이 완료되었습니다. 파일명 목록이 '{log_file_path}'에 저장되었습니다.")


if __name__ == "__main__":
    # --- 사용 방법 ---
    # 1. 원본 이미지가 있는 폴더 경로를 지정합니다. (프로젝트 루트 기준)
    input_directory = "dataset_split/test/input"

    # 2. 리사이즈된 이미지를 저장할 폴더 경로를 지정합니다.
    output_directory = "processdata/input"

    # 3. 목표 해상도를 지정합니다 (너비, 높이 순).
    width = 360
    height = 240

    # 4. 파일명 목록을 저장할 txt 파일의 전체 경로와 이름을 지정합니다.
    # [수정] 다른 스크립트들과의 호환성을 위해 preprocessing 폴더 안에 저장합니다.
    log_file = "preprocessing/resized_image_list.txt"

    # 5. 함수를 실행합니다.
    batch_resize_and_log_images(input_directory, output_directory, width, height, log_file)