import os
import shutil
import random

def split_dataset(base_dir=".", output_dir="split_data2", list_file="resized_image_list.txt"):
    """
    데이터셋을 train, test, validation 세트로 분할합니다.
    """
    # 1. 경로 설정
    source_data_dir = os.path.join(base_dir, "processdata")
    source_gt_dir = os.path.join(source_data_dir, "gt")
    source_input_dir = os.path.join(source_data_dir, "input")
    file_list_path = os.path.join(base_dir, list_file)
    
    if not os.path.isdir(source_data_dir):
        print(f"오류: 소스 데이터 폴더 '{source_data_dir}'를 찾을 수 없습니다.")
        return
    if not os.path.exists(file_list_path):
        print(f"오류: 파일 목록 '{file_list_path}'를 찾을 수 없습니다.")
        return

    print("===== 데이터셋 분할을 시작합니다 =====")
    print(f"기준 파일: {file_list_path}")

    # 2. 파일 목록 읽기 및 파일명만 추출  <- 이 부분이 수정되었습니다!
    with open(file_list_path, 'r') as f:
        # 각 줄에서 경로를 읽어온 뒤, os.path.basename()을 사용해 파일명만 추출합니다.
        filenames = [os.path.basename(line.strip()) for line in f if line.strip()]
    
    # 파일 목록을 무작위로 섞음
    random.shuffle(filenames)
    
    total_files = len(filenames)
    print(f"총 파일 수: {total_files}개")

    # 3. 비율에 따라 분할 인덱스 계산
    train_ratio = 0.7
    test_ratio = 0.2
    
    train_end_idx = int(total_files * train_ratio)
    test_end_idx = train_end_idx + int(total_files * test_ratio)

    train_files = filenames[:train_end_idx]
    test_files = filenames[train_end_idx:test_end_idx]
    val_files = filenames[test_end_idx:]

    print(f"분할 결과: Train({len(train_files)}), Test({len(test_files)}), Val({len(val_files)})")

    split_map = {
        'train': train_files,
        'test': test_files,
        'val': val_files
    }
    
    # 4. 새로운 폴더 구조 생성 및 파일 복사
    print(f"\n'{output_dir}' 폴더에 파일 복사를 시작합니다...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, file_list in split_map.items():
        target_gt_dir = os.path.join(output_dir, split_name, 'gt')
        target_input_dir = os.path.join(output_dir, split_name, 'input')
        
        os.makedirs(target_gt_dir, exist_ok=True)
        os.makedirs(target_input_dir, exist_ok=True)
        
        print(f" -> '{split_name}' 세트 복사 중...")
        
        for filename in file_list:
            src_gt_path = os.path.join(source_gt_dir, filename)
            src_input_path = os.path.join(source_input_dir, filename)
            
            dst_gt_path = os.path.join(target_gt_dir, filename)
            dst_input_path = os.path.join(target_input_dir, filename)
            
            if os.path.exists(src_gt_path):
                shutil.copy2(src_gt_path, dst_gt_path)
            if os.path.exists(src_input_path):
                shutil.copy2(src_input_path, dst_input_path)

    print("\n===== 모든 파일 복사가 완료되었습니다! =====")


if __name__ == "__main__":
    split_dataset(base_dir=".", output_dir="dataset2_split", list_file="resized_image_list.txt")