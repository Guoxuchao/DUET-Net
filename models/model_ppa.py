import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import torchvision.ops

# ==========================================
# Part 0: Configuration and Helper Functions
# ==========================================

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# ==========================================
# Part 1: PPA Module (Core Innovation Component)
# ==========================================

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            
        B, H_pad, W_pad, _ = x.shape
        
        local_patches = x.unfold(1, P, P).unfold(2, P, P)
        local_patches = local_patches.reshape(B, -1, P * P, C)
        local_patches = local_patches.mean(dim=-1)

        local_patches = self.mlp1(local_patches)
        local_patches = self.norm(local_patches)
        local_patches = self.mlp2(local_patches)

        local_attention = F.softmax(local_patches, dim=-1)
        local_out = local_patches * local_attention

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        local_out = local_out.reshape(B, H_pad // P, W_pad // P, self.output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

class conv_block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), 
                 dilation=(1, 1), norm_type='bn', activation=True, use_bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=use_bias, groups=groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()
        self.skip = conv_block(in_features=in_features, out_features=filters, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)
        self.c1 = conv_block(in_features=in_features, out_features=filters, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c2 = conv_block(in_features=filters, out_features=filters, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c3 = conv_block(in_features=filters, out_features=filters, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)
        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UncertaintyBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        # 1. Depth-wise Conv (5x5) - Captures local spatial info
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # 2. Depth-wise Dilated Conv (7x7, d=3) - Captures long-range spatial info
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        # 3. Point-wise Conv (1x1) - Channel adaptation
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn # Attention mechanism: feature * attention_map

class ContextBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ContextBlock, self).__init__()
        inter_channels = out_channels // 4
        
        # Branch 1: 1x1 Conv (Pixel-level)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 Conv (Small scale)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 Conv (Medium scale)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 Dilated(3) (Large scale, equiv 7x7)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

# ==========================================
# Part 2: Innovative Model (VGG + FPN + PPA + Multi-Branch Head)
# ==========================================

model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class UncertaintyVGG19_FPN_PPA_MultiBranch(nn.Module):
    """
    Multi-Branch VGG19_FPN_PPA Model with Uncertainty Estimation
    Structure:
    1. VGG19 Backbone + FPN + PPA (Shared Feature Extractor)
    2. Main Branch (Regression): Direct regression of density map (1 channel)
    3. Aux Branch 1 (Classification): Multi-scale classification (25 channels)
    4. Aux Branch 2 (Uncertainty): Captures contextual uncertainty (1 channel)
    """
    def __init__(self, features, use_ppa=True, use_cls_head=True, use_unc_head=True):
        super(UncertaintyVGG19_FPN_PPA_MultiBranch, self).__init__()
        self.features = features
        self.use_ppa = use_ppa
        self.use_cls_head = use_cls_head
        self.use_unc_head = use_unc_head
        
        # ====================
        # Shared Feature Extractor
        # ====================
        # PPA Module (Optional)
        if self.use_ppa:
            self.ppa_deep = PPA(in_features=512, filters=512)
        
        # FPN Layers
        self.lat_layer2 = nn.Conv2d(512, 256, 1)
        self.lat_layer3 = nn.Conv2d(512, 256, 1)
        self.smooth1 = nn.Conv2d(256, 128, 3, padding=1)
        
        # ====================
        # Branch 1: Main Regression Head (Density Map)
        # ====================
        # Input: FPN Feature (128 ch) -> Output: Density (1 ch)
        if self.use_ppa:
            self.ppa_reg = PPA(in_features=128, filters=128) 
            
        self.head_reg = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True) # Non-negative density map
        )
        
        # ====================
        # Branch 2: Classification Head
        # ====================
        if self.use_cls_head:
            self.mscb_branch = ContextBlock(in_channels=128, out_channels=128)
            self.head_cls = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 10, 1), # 10 density level logits
                nn.Softmax(dim=1) 
            )
        
        # ====================
        # Branch 3: Uncertainty Head
        # ====================
        if self.use_unc_head:
            self.lka_branch = UncertaintyBlock(dim=128)
            self.head_unc = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Softplus() 
            )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Backbone + Neck (Shared)
        x_feat1 = self.features[:19](x)
        x_feat2 = self.features[19:27](x_feat1) # 1/8 scale
        x_feat3 = self.features[27:36](x_feat2) # 1/16 scale
        
        # PPA Enhancement
        if self.use_ppa:
            p3 = self.ppa_deep(x_feat3)
        else:
            p3 = x_feat3 
            
        p3_lat = self.lat_layer3(p3)
        
        # FPN Fusion
        p2 = self.lat_layer2(x_feat2) + F.interpolate(p3_lat, scale_factor=2, mode='nearest', recompute_scale_factor=False)
        final_feat = self.smooth1(p2) 
        
        # 2. Main Branch: Regression
        if self.use_ppa:
            feat_reg = self.ppa_reg(final_feat)
            feat_reg = feat_reg + final_feat
        else:
            feat_reg = final_feat
            
        density_map = self.head_reg(feat_reg) 
        
        # 3. Aux Branch 1: Classification
        logits = None
        if self.use_cls_head:
            feat_cls = self.mscb_branch(final_feat)
            logits = self.head_cls(feat_cls) 
        
        # 4. Aux Branch 2: Uncertainty
        uncertainty_map = None
        if self.use_unc_head:
            feat_unc = self.lka_branch(final_feat)
            uncertainty_map = self.head_unc(feat_unc) 
        
        return density_map, logits, uncertainty_map


