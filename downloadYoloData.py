from pathlib import Path
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# 1. 'yaml' 변수와 'path' 키를 직접 정의해야 합니다.
#    (예: 데이터를 'C:\datasets\coco' 폴더에 저장하려는 경우)
yaml = {
    'path': 'C:\Users\zmffk\OneDrive\바탕 화면\model\yolodata',
}
# 2. 'ultralytics' 라이브러리가 설치되어 있어야 합니다.
#    pip install ultralytics

# --- (사용자가 제공한 코드 시작) ---
segments = True  # segment or box labels
dir = Path(yaml["path"])  # dataset root dir
urls = [ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")]  # labels
download(urls, dir=dir.parent)  # dir.parent는 'C:\datasets'가 됩니다.

# Download data
urls = [
    "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
]
# dir / "images"는 'C:\datasets\coco\images'가 됩니다.
download(urls, dir=dir / "images", threads=3)
# --- (사용자가 제공한 코드 끝) ---

print(f"다운로드 완료! 데이터는 {dir} 경로에 저장되었습니다.")