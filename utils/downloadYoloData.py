from pathlib import Path
import os
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# 1. 경로 설정 (★★★ 프로젝트 구조에 맞게 수정됨 ★★★)
# -------------------------------------------------------------------
# 프로젝트 루트 기준 'data/yolo_data' 폴더에 저장합니다.
# 실행 위치(Project Root)를 기준으로 경로를 잡습니다.
target_dir = Path('data/yolo_data')

# Ultralytics는 절대 경로를 선호하므로 절대 경로로 변환
yaml = {
    'path': str(target_dir.resolve()),
}
# -------------------------------------------------------------------

# 2. 'ultralytics' 라이브러리 체크
# pip install ultralytics

# --- (데이터 다운로드 로직 시작) ---
print(f"다운로드 경로 설정됨: {yaml['path']}")

segments = True  # segment or box labels
dir = Path(yaml["path"])  # dataset root dir

# 라벨 데이터 다운로드
# (coco2017labels-segments.zip 또는 coco2017labels.zip)
urls = [ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")]
# dir.parent에 저장 (즉, data/ 폴더에 압축 해제됨)
download(urls, dir=dir.parent)

# 이미지 데이터 다운로드 (Val2017)
urls = [
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
]
# dir / "images" (즉, data/yolo_data/images)에 저장
download(urls, dir=dir / "images", threads=3)
# --- (데이터 다운로드 로직 끝) ---

print(f"\n✅ 다운로드 완료! 데이터는 '{dir}' 경로에 저장되었습니다.")