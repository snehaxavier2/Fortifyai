import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import timm # type: ignore


class FrequencyBranch(nn.Module):
    
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((16, 16)) 
        self.proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1)                              
        fft  = torch.fft.fft2(gray)                       
        mag  = torch.log(torch.abs(fft) + 1e-8)          
        mag  = self.pool(mag.unsqueeze(1)).squeeze(1)     
        mag  = mag.flatten(1)                              
        return self.proj(mag)                              

class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class HybridModel(nn.Module):

    PERTURBATION_KERNEL = 5

    def __init__(self, pretrained: bool = True, se_reduction: int = 16):
        super().__init__()

        # Backbone 
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained  = pretrained,
            num_classes = 0,
            global_pool = ""
        )
        self.pool        = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1536                              
        self.freq_dim    = 256                               
        self.fusion_dim  = self.feature_dim * 2 + self.freq_dim  

        # Frequency branch 
        self.freq_branch = FrequencyBranch(out_dim=self.freq_dim)

        # SE attention 
        self.se_attention = SEBlock(self.fusion_dim, reduction=se_reduction)

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.set_grad_checkpointing(enable=True)
        print("[HybridModel] Gradient checkpointing enabled.")

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[HybridModel] Backbone frozen — training FFT branch + SE + classifier.")

    def unfreeze_last_blocks(self, n: int = 3) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        start_idx = 7 - n
        for name, module in self.backbone.named_modules():
            if name.startswith("blocks."):
                if int(name.split(".")[1]) >= start_idx:
                    for p in module.parameters():
                        p.requires_grad = True
            if name in ("conv_head", "bn2"):
                for p in module.parameters():
                    p.requires_grad = True
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[HybridModel] Unfroze last {n}/7 blocks + head ({trainable:,} params).")

    def count_parameters(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.backbone(x)), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pixel-domain features 
        f_orig = self.extract_features(x)                     

        x_pert = F.avg_pool2d(
            x,
            kernel_size = self.PERTURBATION_KERNEL,
            stride      = 1,
            padding     = self.PERTURBATION_KERNEL // 2
        )
        f_sub = f_orig - self.extract_features(x_pert)         

        # Frequency-domain features 
        f_freq = self.freq_branch(x)                            

        # Fusion + SE attention + classification 
        fusion = torch.cat([f_orig, f_sub, f_freq], dim=1)    
        fusion = self.se_attention(fusion)
        return self.classifier(fusion)                          


def print_model_summary(model: HybridModel, device: torch.device) -> None:
    params = model.count_parameters()
    print("\n" + "=" * 60)
    print(" HybridModel v5 — FortifyAI (FFT + 224×224)")
    print("=" * 60)
    print(f"  Backbone          : EfficientNet-B3")
    print(f"  Input resolution  : 224 × 224 × 3")
    print(f"  Feature dim       : {model.feature_dim}")
    print(f"  Freq branch dim   : {model.freq_dim}")
    print(f"  Perturbation k    : {model.PERTURBATION_KERNEL}")
    print(f"  Fusion dim        : {model.fusion_dim}  (1536+1536+256)")
    print(f"  Total params      : {params['total']:,}")
    print(f"  Trainable params  : {params['trainable']:,}")

    dummy = torch.zeros(2, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Forward pass (batch=2, input=224×224):")
    print(f"    Input  : {list(dummy.shape)}")
    print(f"    Output : {list(out.shape)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HybridModel(pretrained=True).to(device)
    print_model_summary(model, device)
    model.enable_gradient_checkpointing()
    model.freeze_backbone()
    print(f"  Stage 1 trainable: {model.count_parameters()['trainable']:,}")
    model.unfreeze_last_blocks(n=3)
    print(f"  Stage 2 trainable: {model.count_parameters()['trainable']:,}")
