import os
from tqdm import tqdm

# 1. 경로 설정 (★★★ 사용 전 꼭 수정해주세요 ★★★)
# -------------------------------------------------------------------
# 이름 변경을 할 파일들이 들어있는 'input' 폴더 경로
# 예: D:/processdata/input/
input_folder_path = 'processdata/input'

# 정답 이름(gt 이름) 목록이 들어있는 TXT 파일 경로
# 예: C:/Users/MyUser/Desktop/clean_list.txt
# (수정됨: preprocessing 폴더 내 파일을 바라보도록 변경)
txt_file_path = 'preprocessing/resized_image_list.txt'
# -------------------------------------------------------------------

# 2. 이름 변경 로직
# -------------------------------------------------------------------
# 만약을 위해 원본 파일을 백업해두시는 것을 권장합니다.

print("파일 이름 변경을 시작합니다...")
renamed_count = 0
not_found_count = 0

try:
    # 1. TXT 파일에서 목표 이름들을 모두 읽어와서 세트(set)에 저장 (검색 속도 향상)
    with open(txt_file_path, 'r') as f:
        # 각 줄의 파일명에서 확장자를 제외한 기본 이름만 저장 (예: 'NYU2_1')
        target_basenames = {os.path.splitext(line.strip())[0] for line in f if line.strip()}

    # 2. input 폴더에 있는 모든 파일 목록을 가져옴
    files_in_input_folder = os.listdir(input_folder_path)

    for current_filename in tqdm(files_in_input_folder, desc="Renaming files"):
        # 3. 현재 파일 이름에서 기본 이름 부분을 추출
        # 예: 'NYU2_1_3_1.jpg' -> 'NYU2_1'
        try:
            parts = current_filename.split('_')
            current_basename = f"{parts[0]}_{parts[1]}"
        except IndexError:
            # 이름 형식이 예상과 다를 경우 건너뜀 (예: .DS_Store 파일)
            continue
            
        # 4. 추출한 기본 이름이 목표 이름 목록(세트)에 있는지 확인
        if current_basename in target_basenames:
            # 5. 새로운 이름 결정 (기본 이름 + 원래 확장자)
            original_extension = os.path.splitext(current_filename)[1]
            new_filename = f"{current_basename}{original_extension}"
            
            # 6. 이전 경로와 새 경로 설정
            old_filepath = os.path.join(input_folder_path, current_filename)
            new_filepath = os.path.join(input_folder_path, new_filename)
            
            # 7. 파일 이름 변경 실행
            os.rename(old_filepath, new_filepath)
            renamed_count += 1
        else:
            not_found_count += 1

    print("\n작업이 완료되었습니다!")
    print(f"총 {renamed_count}개의 파일 이름이 변경되었습니다.")
    if not_found_count > 0:
        print(f"{not_found_count}개의 파일은 TXT 파일에 짝이 없어 변경되지 않았습니다.")

except FileNotFoundError:
    print(f"오류: TXT 파일 또는 input 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")

# -------------------------------------------------------------------