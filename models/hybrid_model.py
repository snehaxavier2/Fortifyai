import torch # type: ignore
import torch.nn as nn # type: ignore
import torchvision.models as models # type: ignore


class FFTBranch(nn.Module):
    def __init__(self):
        super(FFTBranch, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x  # [batch, 64]


class HybridModel(nn.Module):
    def __init__(self, pretrained=True):
        super(HybridModel, self).__init__()

        # Spatial Branch
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.spatial_features = mobilenet.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_feature_dim = 1280

        # FFT Branch
        self.fft_branch = FFTBranch()
        self.fft_feature_dim = 64

        # Fusion MLP
        fusion_input_dim = self.spatial_feature_dim + self.fft_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1)
        )

    def forward(self, rgb, fft):

        # Spatial branch
        spatial = self.spatial_features(rgb)
        spatial = self.spatial_pool(spatial)
        spatial = torch.flatten(spatial, 1)

        # FFT branch
        fft_feat = self.fft_branch(fft)

        # Concatenate
        fused = torch.cat((spatial, fft_feat), dim=1)

        # Classification
        output = self.classifier(fused)

        return output