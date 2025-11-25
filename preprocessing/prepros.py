import os
from glob import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RainDSSynDataset(Dataset):
    """
    Ground Truth 이미지와 비가 오는 이미지를 쌍으로 불러오는 PyTorch Dataset 클래스입니다.

    - Args:
        root_gt (str): Ground Truth 이미지(.jpg, .png)가 있는 폴더 경로
        root_rain (str): 비 오는 이미지가 있는 폴더 경로
        img_size (tuple): 이미지 크기를 조절할 (너비, 높이)
        transform (callable, optional): 이미지에 적용할 사용자 정의 transform. 
                                        None이면 기본 transform(Resize, ToTensor)이 적용됩니다.
    """
    def __init__(self,
                 root_gt: str = "./dataset_split/train/gt",
                 root_rain: str = "./dataset_split/train/input",
                 img_size: Tuple[int, int] = (480, 720), # (높이, 너비) 순서로 지정
                 transform=None):
        super().__init__()
        print("[Dataset] 초기화 중...")

        self.root_gt = root_gt
        self.root_rain = root_rain
        self.img_size = img_size
        
        # 1. GT 이미지 경로 목록 가져오기
        gt_patterns = ["*.jpg", "*.png"]
        self.gt_paths = []
        for pat in gt_patterns:
            self.gt_paths.extend(glob(os.path.join(self.root_gt, pat)))
        
        # 중복 제거 및 정렬
        self.gt_paths = sorted(list(set(self.gt_paths)))

        if not self.gt_paths:
            raise RuntimeError(f"[Dataset] GT 이미지가 없습니다: {self.root_gt}")
        print(f"[Dataset] GT 총 이미지 개수: {len(self.gt_paths)}")

        # 2. (최적화) Rain 파일 목록을 '한 번만' 불러와 Set으로 만듭니다.
        # Set은 리스트보다 특정 항목을 찾는 속도가 월등히 빠릅니다.
        print("[Dataset] Rain 파일 목록을 미리 불러오는 중...")
        rain_patterns = ["*.jpg", "*.png"]
        all_rain_paths = []
        for pat in rain_patterns:
            all_rain_paths.extend(glob(os.path.join(self.root_rain, pat)))
        self.rain_filenames_set = {os.path.basename(p) for p in all_rain_paths}
        print(f"[Dataset] Rain 총 파일 개수: {len(self.rain_filenames_set)}")

        # 3. (최적화) 디스크 I/O 없이 메모리에서 파일 쌍 매칭
        self.pairs = []
        self.unmatched_gts = 0
        print("[Dataset] GT 파일과 Rain 파일 쌍 매칭 중 (메모리 기반)...")
        for gt_path in self.gt_paths:
            fname = os.path.basename(gt_path)
            if fname in self.rain_filenames_set:
                rain_path = os.path.join(self.root_rain, fname)
                self.pairs.append((rain_path, gt_path))
            else:
                self.unmatched_gts += 1
        
        if self.unmatched_gts > 0:
            print(f"[Dataset] 경고: 짝을 찾지 못한 GT 파일이 {self.unmatched_gts}개 있습니다.")

        if not self.pairs:
            raise RuntimeError(f"[Dataset] 매칭되는 GT-Rain 쌍이 없습니다. 폴더를 확인하세요.")
        
        print(f"\n[Dataset] 총 매칭 쌍 개수: {len(self.pairs)}")
        print("[Dataset] 초기화 완료\n")

        # 4. Transform 설정
        if transform is None:
            print("[Dataset] 기본 transform (Resize → ToTensor) 설정")
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
        else:
            print("[Dataset] 사용자 지정 transform 사용")
            self.transform = transform

    def __len__(self):
        """데이터셋의 총 쌍 개수를 반환합니다."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """인덱스(idx)에 해당하는 이미지 쌍(rain, gt)을 불러와 반환합니다."""
        rain_path, gt_path = self.pairs[idx]

        try:
            # 이미지를 열고 RGB로 변환
            gt_img = Image.open(gt_path).convert("RGB")
            rain_img = Image.open(rain_path).convert("RGB")

            # transform 적용
            gt_t = self.transform(gt_img)
            rain_t = self.transform(rain_img)

            return {
                "rain": rain_t,
                "gt": gt_t,
                "rain_path": rain_path,
                "gt_path": gt_path
            }
        except Exception as e:
            print(f"[Dataset] 오류: 이미지 로딩/처리 중 오류 발생 ({os.path.basename(rain_path)}) - {e}")
            # 오류 발생 시 None을 반환하도록 처리 (DataLoader에서 걸러낼 수 있도록)
            return None


def collate_fn(batch):
    """
    __getitem__에서 None이 반환되었을 경우, 해당 데이터를 배치에서 제외합니다.
    DataLoader의 collate_fn 인자로 사용됩니다.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None