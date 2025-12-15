import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class PSNRLoss(nn.Module):
    def __init__(self, data_range=1.0, normalize=True):
        super(PSNRLoss, self).__init__()
        self.data_range = data_range
        self.normalize = normalize

    def forward(self, pred, target):
        if self.normalize:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(0.0, device=pred.device) # Avoid log(0)
        psnr = 10 * torch.log10((self.data_range ** 2) / mse)
        return -psnr # Minimize negative PSNR
    
class SSIMLoss(nn.Module):
    def __init__(self, channel=1, normalize=True):
        super(SSIMLoss, self).__init__()
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=channel)
        self.normalize = normalize
    
    def forward(self, pred, target):
        if self.normalize:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        ssim_value = self.ssim_module(pred, target)
        return 1 - ssim_value  # Minimize (1 - SSIM)