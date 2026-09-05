import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class Losses(nn.Module):
    def __init__(self, device):
        super(Losses, self).__init__()
        self.device = device
        self.ssim_window_size = 9
        self.ssim_window = create_window(self.ssim_window_size, 1).to(self.device)

    def ssim_loss(self, img1, img2):
        return 1.0 - _ssim(img1, img2, self.ssim_window, self.ssim_window_size, 1)

    def masked_ssim_loss(self, img1, img2, mask):
        window = self.ssim_window
        window_size = self.ssim_window_size
        channel = 1
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        loss_map = 1.0 - ssim_map
        loss = torch.sum(loss_map * mask) / (torch.sum(mask) + 1e-7)
        return loss

    def ranking_consistency_loss(self, s_density, t_density, mask_drf):
        B, C, H, W = s_density.shape
        s_flat = s_density.view(B, -1)
        t_flat = t_density.view(B, -1)
        mask_flat = mask_drf.view(B, -1)
        
        loss_rank_batch = 0.0
        
        for b in range(B):
            valid_indices = torch.nonzero(mask_flat[b] > 0.5, as_tuple=True)[0]
            
            if len(valid_indices) < 100:
                continue
                
            num_samples = 2000
            num_valid = len(valid_indices)
            
            if num_valid > num_samples:
                perm = torch.randperm(num_valid, device=self.device)[:num_samples]
                indices = valid_indices[perm]
            else:
                indices = valid_indices
                
            s_sampled = s_flat[b, indices]
            t_sampled = t_flat[b, indices]
            
            s_i = s_sampled.unsqueeze(1)
            s_j = s_sampled.unsqueeze(0)
            
            t_i = t_sampled.unsqueeze(1)
            t_j = t_sampled.unsqueeze(0)
            
            target_direction = torch.sign(t_i - t_j)
            
            diff_t = torch.abs(t_i - t_j)
            threshold = 0.001 
            valid_pairs = diff_t > threshold
            
            if torch.sum(valid_pairs) < 1:
                continue
                
            loss_pairs = F.relu(-target_direction * (s_i - s_j))
            loss_rank = torch.sum(loss_pairs * valid_pairs) / (torch.sum(valid_pairs) + 1e-7)
            loss_rank_batch += loss_rank
            
        return loss_rank_batch / B

    def consistency_loss(self, s_density, t_density, mask_drf):
        loss_mae = torch.abs(s_density - t_density)
        loss_mae = torch.sum(loss_mae * mask_drf) / (torch.sum(mask_drf) + 1e-7)
        return 100.0 * loss_mae

    def classification_consistency_loss(self, s_probs, t_probs, mask_drf):
        if s_probs is None or t_probs is None:
            return 0.0

        loss_mae = torch.abs(s_probs - t_probs)
        loss_mae = torch.sum(loss_mae, dim=1, keepdim=True)
        
        loss = torch.sum(loss_mae * mask_drf) / (torch.sum(mask_drf) + 1e-7)
        return loss

    def uncertainty_consistency_loss(self, s_logvar, t_logvar, mask_drf):
        if s_logvar is None or t_logvar is None:
            return 0.0
            
        loss_mse = (s_logvar - t_logvar) ** 2
        loss = torch.sum(loss_mse * mask_drf) / (torch.sum(mask_drf) + 1e-7)
        return loss
