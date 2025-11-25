import os
import glob
import random
import shutil
from tqdm import tqdm

# 1. 경로 및 파일 설정 (★★★ 사용 전 꼭 수정해주세요 ★★★)
# -------------------------------------------------------------------
# 안개 낀 원본 이미지가 있는 폴더 경로
# 예: D:/datasets/NYU2_hazy/
hazy_data_path = 'data/data'

# 최종적으로 안개 이미지를 저장할 'input' 폴더 경로
# 예: D:/processdata/input/
save_input_path = 'processdata/input'

# 깨끗한 이미지 파일 이름이 저장된 TXT 파일 경로
# 예: C:/Users/MyUser/Desktop/clean_list.txt
txt_file_path = 'resized_image_list.txt'
# -------------------------------------------------------------------

# 2. 저장 폴더 준비
# -------------------------------------------------------------------
if not os.path.exists(save_input_path):
    os.makedirs(save_input_path)
    print(f"'{save_input_path}' 폴더를 생성했습니다.")
else:
    print(f"'{save_input_path}' 폴더의 기존 내용을 삭제하고 새로 시작합니다.")
    for f in os.listdir(save_input_path):
        os.remove(os.path.join(save_input_path, f))
# -------------------------------------------------------------------

# 3. 데이터 선택 및 복사 로직
# -------------------------------------------------------------------
print("데이터 선택 및 복사를 시작합니다...")

try:
    with open(txt_file_path, 'r') as f:
        image_filenames = [line.strip() for line in f if line.strip()]

    unique_filenames = sorted(list(set(image_filenames)))

    for filename in tqdm(unique_filenames, desc="Selecting hazy images"):
        # ★★★ 수정된 부분 ★★★
        # 파일명에서 확장자(.jpg)를 분리하여 기본 이름만 사용합니다.
        # 예: 'NYU2_1047.jpg' -> 'NYU2_1047'
        basename = os.path.splitext(filename)[0]
        
        # 안개 이미지 폴더에서 현재 이름에 해당하는 모든 파일 찾기
        search_pattern = os.path.join(hazy_data_path, f"{basename}_*.jpg")
        hazy_files = glob.glob(search_pattern)

        if hazy_files:
            # 찾은 안개 이미지 리스트 중에서 하나를 무작위로 선택
            random_hazy_file = random.choice(hazy_files)
            
            # 선택된 파일을 'input' 폴더로 바로 복사
            shutil.copy(random_hazy_file, save_input_path)
        else:
            print(f"경고: '{basename}_*.jpg' 패턴에 해당하는 파일을 찾을 수 없습니다.")

    print("\n작업이 완료되었습니다!")
    print(f"총 {len(os.listdir(save_input_path))}개의 안개 이미지가 '{save_input_path}'에 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: TXT 파일을 찾을 수 없습니다. 경로를 확인해주세요: {txt_file_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
# -------------------------------------------------------------------