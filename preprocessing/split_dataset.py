import os
import shutil
import random

def split_dataset(base_dir=".", output_dir="dataset_split", list_file_path="preprocessing/resized_image_list.txt"):
    """
    데이터셋을 train, test, validation 세트로 분할합니다.
    
    :param base_dir: 프로젝트 루트 경로 (기본: 현재 폴더)
    :param output_dir: 분할된 데이터가 저장될 폴더명 (기본: dataset_split)
    :param list_file_path: 파일 목록이 담긴 txt 파일 경로 (기본: preprocessing/resized_image_list.txt)
    """
    # 1. 경로 설정 (프로젝트 루트 기준)
    # processdata 폴더 안에 gt와 input 폴더가 있다고 가정합니다.
    source_data_dir = os.path.join(base_dir, "processdata")
    source_gt_dir = os.path.join(source_data_dir, "gt")
    source_input_dir = os.path.join(source_data_dir, "input")
    
    # 텍스트 파일 전체 경로
    file_list_full_path = os.path.join(base_dir, list_file_path)
    
    # 경로 유효성 검사
    if not os.path.isdir(source_data_dir):
        print(f"🚨 오류: 소스 데이터 폴더 '{source_data_dir}'를 찾을 수 없습니다.")
        print("   (참고: datatest.py 또는 resize.py를 먼저 실행하여 processdata를 준비해야 합니다.)")
        return
    if not os.path.exists(file_list_full_path):
        print(f"🚨 오류: 파일 목록 '{file_list_full_path}'를 찾을 수 없습니다.")
        return

    print("===== 데이터셋 분할을 시작합니다 =====")
    print(f"기준 파일: {file_list_full_path}")

    # 2. 파일 목록 읽기 및 파일명만 추출
    try:
        with open(file_list_full_path, 'r') as f:
            # 각 줄에서 경로를 읽어온 뒤, os.path.basename()을 사용해 파일명만 추출합니다.
            filenames = [os.path.basename(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return

    # 파일 목록을 무작위로 섞음
    random.shuffle(filenames)
    
    total_files = len(filenames)
    print(f"총 파일 수: {total_files}개")
    
    if total_files == 0:
        print("처리할 파일이 없습니다.")
        return

    # 3. 비율에 따라 분할 인덱스 계산 (Train: 70%, Test: 20%, Val: 10%)
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
    final_output_dir = os.path.join(base_dir, output_dir)
    print(f"\n'{final_output_dir}' 폴더에 파일 복사를 시작합니다...")
    
    # 기존 폴더가 있다면 덮어쓰거나 그대로 진행 (os.makedirs exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    for split_name, file_list in split_map.items():
        target_gt_dir = os.path.join(final_output_dir, split_name, 'gt')
        target_input_dir = os.path.join(final_output_dir, split_name, 'input')
        
        os.makedirs(target_gt_dir, exist_ok=True)
        os.makedirs(target_input_dir, exist_ok=True)
        
        print(f" -> '{split_name}' 세트 복사 중...")
        
        for filename in file_list:
            src_gt_path = os.path.join(source_gt_dir, filename)
            src_input_path = os.path.join(source_input_dir, filename)
            
            dst_gt_path = os.path.join(target_gt_dir, filename)
            dst_input_path = os.path.join(target_input_dir, filename)
            
            # 파일이 실제로 존재할 때만 복사
            if os.path.exists(src_gt_path):
                shutil.copy2(src_gt_path, dst_gt_path)
            if os.path.exists(src_input_path):
                shutil.copy2(src_input_path, dst_input_path)

    print("\n===== 모든 파일 복사가 완료되었습니다! =====")


if __name__ == "__main__":
    # 프로젝트 루트에서 실행한다고 가정
    # 기본값: processdata 폴더를 읽어서 dataset_split 폴더로 분할
    split_dataset(
        base_dir=".", 
        output_dir="dataset_split", 
        list_file_path="preprocessing/resized_image_list.txt"
    )