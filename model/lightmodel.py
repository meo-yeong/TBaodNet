import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================
# 경량화된 모델 정의: DerainNetLite
# ================================================
class DerainNetLite(nn.Module):
    def __init__(self):
        super(DerainNetLite, self).__init__()
        print("[Model] DerainNetLite 초기화 중...")

        # ----- K1 브랜치 (채널: 16 -> 8, 3-conv 블록 -> 2-conv) -----
        self.k1_conv_d1_1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k1_bn_d1_1   = nn.BatchNorm2d(8)
        self.k1_conv_d1_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k1_bn_d1_2   = nn.BatchNorm2d(8)
        # 3번째 conv 레이어 제거

        self.k1_conv_d2_1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k1_bn_d2_1   = nn.BatchNorm2d(8)
        self.k1_conv_d2_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k1_bn_d2_2   = nn.BatchNorm2d(8)

        self.k1_conv_d3_1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k1_bn_d3_1   = nn.BatchNorm2d(8)
        self.k1_conv_d3_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k1_bn_d3_2   = nn.BatchNorm2d(8)

        # 융합 레이어의 입력 채널도 축소 (8 * 3 = 24)
        self.k1_fuse_conv1 = nn.Conv2d(8 * 3, 16, kernel_size=1, stride=1, padding=0)
        self.k1_bn_fuse1   = nn.BatchNorm2d(16)
        self.k1_fuse_conv2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.k1_bn_fuse2   = nn.BatchNorm2d(8)

        self.k1_out = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)

        # ----- K2 브랜치 (채널: 8->4, 16->8, 32->16, 3-conv 블록 -> 2-conv) -----
        self.k2_conv_d0 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k2_bn_d0   = nn.BatchNorm2d(4)

        self.k2_conv_d1_1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k2_bn_d1_1   = nn.BatchNorm2d(8)
        self.k2_conv_d1_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.k2_bn_d1_2   = nn.BatchNorm2d(16)
        # 3번째 conv 레이어 제거

        self.k2_conv_d2_1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k2_bn_d2_1   = nn.BatchNorm2d(8)
        self.k2_conv_d2_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k2_bn_d2_2   = nn.BatchNorm2d(16)
        # 3번째 conv 레이어 제거

        self.k2_conv_d3_1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k2_bn_d3_1   = nn.BatchNorm2d(8)
        self.k2_conv_d3_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=3, dilation=3)
        self.k2_bn_d3_2   = nn.BatchNorm2d(16)
        # 3번째 conv 레이어 제거

        # 융합 레이어의 입력 채널도 축소 (16 * 3 = 48)
        self.k2_fuse_conv1 = nn.Conv2d(16 * 3, 16, kernel_size=1, stride=1, padding=0)
        self.k2_bn_fuse1   = nn.BatchNorm2d(16)
        self.k2_fuse_conv2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.k2_bn_fuse2   = nn.BatchNorm2d(8)

        self.k2_out = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)

        # ----- K3 브랜치 (채널: 16 -> 8) -----
        self.k3_conv_d0 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k3_bn_d0   = nn.BatchNorm2d(8)

        self.k3_conv_d1_1= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k3_bn_d1_1  = nn.BatchNorm2d(8)
        self.k3_conv_d1_2= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.k3_bn_d1_2  = nn.BatchNorm2d(8)

        self.k3_conv_d2_1= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=4, dilation=4)
        self.k3_bn_d2_1  = nn.BatchNorm2d(8)
        self.k3_conv_d2_2= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=4, dilation=4)
        self.k3_bn_d2_2  = nn.BatchNorm2d(8)

        self.k3_conv_d3_1= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=8, dilation=8)
        self.k3_bn_d3_1  = nn.BatchNorm2d(8)
        self.k3_conv_d3_2= nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=8, dilation=8)
        self.k3_bn_d3_2  = nn.BatchNorm2d(8)

        # 융합 레이어의 입력 채널도 축소 (8 * 3 = 24)
        self.k3_fuse_conv1 = nn.Conv2d(8 * 3, 16, kernel_size=1, stride=1, padding=0)
        self.k3_bn_fuse1   = nn.BatchNorm2d(16)
        self.k3_fuse_conv2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.k3_bn_fuse2   = nn.BatchNorm2d(8)

        self.k3_out = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)

        self.negative_slope = 0.2
        print("[Model] DerainNetLite 초기화 완료\n")

    def forward(self, x):
        # ========== K1 브랜치 ==========
        f1_1 = F.leaky_relu(self.k1_bn_d1_1(self.k1_conv_d1_1(x)), negative_slope=self.negative_slope)
        f1_2 = F.leaky_relu(self.k1_bn_d1_2(self.k1_conv_d1_2(f1_1)), negative_slope=self.negative_slope)
        # f1_3 제거됨
        f1_f = torch.cat([f1_1, f1_2], dim=1) # (B, 16, H, W)

        f2_1 = F.leaky_relu(self.k1_bn_d2_1(self.k1_conv_d2_1(f1_f)), negative_slope=self.negative_slope)
        f2_2 = F.leaky_relu(self.k1_bn_d2_2(self.k1_conv_d2_2(f2_1)), negative_slope=self.negative_slope)
        f2_f = torch.cat([f2_1, f2_2], dim=1) # (B, 16, H, W)

        f3_1 = F.leaky_relu(self.k1_bn_d3_1(self.k1_conv_d3_1(f2_f)), negative_slope=self.negative_slope)
        f3_2 = F.leaky_relu(self.k1_bn_d3_2(self.k1_conv_d3_2(f3_1)), negative_slope=self.negative_slope)
        
        # f1_3 대신 f1_2를 사용하여 융합
        fuse1 = torch.cat([f1_2, f2_2, f3_2], dim=1)
        fuse1 = F.relu(self.k1_bn_fuse1(self.k1_fuse_conv1(fuse1)))
        fuse1 = F.relu(self.k1_bn_fuse2(self.k1_fuse_conv2(fuse1)))
        K1 = self.k1_out(fuse1)

        # ========== K2 브랜치 ==========
        g0 = F.leaky_relu(self.k2_bn_d0(self.k2_conv_d0(x)), negative_slope=self.negative_slope)

        g1 = F.leaky_relu(self.k2_bn_d1_1(self.k2_conv_d1_1(g0)), negative_slope=self.negative_slope)
        g1 = F.leaky_relu(self.k2_bn_d1_2(self.k2_conv_d1_2(g1)), negative_slope=self.negative_slope)
        # g1의 3번째 conv 제거됨

        g2 = F.leaky_relu(self.k2_bn_d2_1(self.k2_conv_d2_1(g0)), negative_slope=self.negative_slope)
        g2 = F.leaky_relu(self.k2_bn_d2_2(self.k2_conv_d2_2(g2)), negative_slope=self.negative_slope)
        # g2의 3번째 conv 제거됨

        g3 = F.leaky_relu(self.k2_bn_d3_1(self.k2_conv_d3_1(g0)), negative_slope=self.negative_slope)
        g3 = F.leaky_relu(self.k2_bn_d3_2(self.k2_conv_d3_2(g3)), negative_slope=self.negative_slope)
        # g3의 3번째 conv 제거됨

        fuse2 = torch.cat([g1, g2, g3], dim=1)
        fuse2 = F.relu(self.k2_bn_fuse1(self.k2_fuse_conv1(fuse2)))
        fuse2 = F.relu(self.k2_bn_fuse2(self.k2_fuse_conv2(fuse2)))
        K2 = self.k2_out(fuse2)

        # ========== K3 브랜치 ==========
        h0 = F.leaky_relu(self.k3_bn_d0(self.k3_conv_d0(x)), negative_slope=self.negative_slope)

        h1_1 = F.leaky_relu(self.k3_bn_d1_1(self.k3_conv_d1_1(h0)), negative_slope=self.negative_slope)
        h1_2 = F.leaky_relu(self.k3_bn_d1_2(self.k3_conv_d1_2(h1_1)), negative_slope=self.negative_slope)

        h2_1 = F.leaky_relu(self.k3_bn_d2_1(self.k3_conv_d2_1(h0)), negative_slope=self.negative_slope)
        h2_2 = F.leaky_relu(self.k3_bn_d2_2(self.k3_conv_d2_2(h2_1)), negative_slope=self.negative_slope)

        h3_1 = F.leaky_relu(self.k3_bn_d3_1(self.k3_conv_d3_1(h0)), negative_slope=self.negative_slope)
        h3_2 = F.leaky_relu(self.k3_bn_d3_2(self.k3_conv_d3_2(h3_1)), negative_slope=self.negative_slope)
        
        fuse3 = torch.cat([h1_2, h2_2, h3_2], dim=1)
        fuse3 = F.relu(self.k3_bn_fuse1(self.k3_fuse_conv1(fuse3)))
        fuse3 = F.relu(self.k3_bn_fuse2(self.k3_fuse_conv2(fuse3)))
        K3 = self.k3_out(fuse3)

        # ========== 최종 복원 식 ==========
        diff = K1 - K2 - K3
        clean = diff * x - diff

        return clean

# 모델 테스트
if __name__ == '__main__':
    # 모델 인스턴스 생성
    model = DerainNetLite()
    
    # 모델 구조 확인
    print(model)
    
    # 임의의 입력 데이터 생성 (Batch=1, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # 모델 추론 테스트
    output = model(dummy_input)
    
    # 출력 텐서의 크기 확인
    print(f"\n입력 텐서 크기: {dummy_input.shape}")
    print(f"출력 텐서 크기: {output.shape}")