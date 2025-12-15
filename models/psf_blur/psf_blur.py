import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import networks


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()

def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window

import torch
import torch.fft as fft
import torch.nn.functional as F


def generate_psf(N, wavelength, NA, refr_index, z_defocus_list, device='cuda'):
    """
    Generate PSFs for a list of defocus values.
    Args:
        N:             image size (NxN)
        wavelength:    wavelength in meters
        NA:            numerical aperture
        refr_index:    refractive index
        z_defocus_list: list or tensor of z values (meters)
    Returns: PSFs: (B, 1, N, N)
    """
    # batch size = number of z values
    z_defocus_list = torch.tensor(z_defocus_list, device=device).float()
    B = len(z_defocus_list)

    # Pupil coordinates (rho)
    x = torch.linspace(-1, 1, N, device=device)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    rho = torch.sqrt(xx**2 + yy**2)

    # Circular aperture
    aperture = (rho <= 1).float()  # 1 inside aperture

    # Wave number
    k = 2 * torch.pi / wavelength

    # Defocus phase term
    # sqrt(1 - (NA/n)^2 * rho^2) is approximated by 1 - rho^2 * NA^2/(2n^2)
    # More accurate version:
    phase = torch.sqrt(1 - (NA/refr_index)**2 * rho**2)

    # Expand for batch
    phase = phase.unsqueeze(0)  # (1, N, N)
    aperture = aperture.unsqueeze(0)        # (1, N, N)

    z = z_defocus_list.view(B, 1, 1)        # (B, 1, 1)

    pupil = aperture * torch.exp(1j * k * z * phase)    # (B, N, N)

    # Fourier transform to get PSF
    psf = fft.fftshift(fft.fft2(pupil), dim=(-2, -1))
    psf = torch.abs(psf) ** 2

    # Normalize
    psf = psf / psf.view(B, -1).sum(dim=1).view(B, 1, 1)

    return psf.unsqueeze(1)  # (B, 1, N, N)

def crop_psf(psf, k):
    """
    psf: (B, 1, N, N)
    returns cropped psf: (B, 1, k, k)
    """
    B, _, N, _ = psf.shape
    center = N // 2
    half = k // 2
    return psf[:, :, center-half:center+half+1,
                      center-half:center+half+1]

class PSFBlurLayer(nn.Module):
    def __init__(self, num_kernels=11, max_kernel_size=255, mode='TG', channels=3):
        super(PSFBlurLayer, self).__init__()
        self.channels = channels
        self.padding = int((max_kernel_size - 1) / 2)
        if self.channels == 1:
            weight = generate_psf(max_kernel_size, wavelength=550e-9, NA=0.65, refr_index=1.0,
                                   z_defocus_list=list(np.arange(0, num_kernels) * 2e-6), device='cuda')
        elif self.channels == 3:
            weight_r = generate_psf(max_kernel_size, wavelength=700e-9, NA=0.65, refr_index=1.0,
                                   z_defocus_list=list(np.arange(0, num_kernels) * 2e-6), device='cuda')
            weight_g = generate_psf(max_kernel_size, wavelength=550e-9, NA=0.65, refr_index=1.0,
                                   z_defocus_list=list(np.arange(0, num_kernels) * 2e-6), device='cuda')
            weight_b = generate_psf(max_kernel_size, wavelength=450e-9, NA=0.65, refr_index=1.0,
                                   z_defocus_list=list(np.arange(0, num_kernels) * 2e-6), device='cuda')
            weight = torch.cat([weight_r, weight_g, weight_b], dim=1)
        # weight = crop_psf(weight, k=max_kernel_size)
        # weight = weight / weight.sum(dim=(2,3), keepdim=True)
        if mode == 'TG':
            self.weight = weight
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels, channels, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = weight
            self.weight.requires_grad = False

    def __call__(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        x = F.conv2d(x, self.weight, padding=0, groups=self.channels)
        return x
    
class GaussianBlurLayer(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        kernel_size = 3
        weight = torch.zeros(num_kernels, 1, max_kernel_size, max_kernel_size)
        for i in range(num_kernels-1):
            pad = int((max_kernel_size - kernel_size) / 2)
            weight[i + 1] = (F.pad(gen_gaussian_kernel(kernel_size, sigma=0.25 * (i + 1)).cuda(),
                                   [pad, pad, pad, pad])).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)

        kernel = np.repeat(weight, self.channels, axis=0).cuda()
        if mode == 'TG':
            self.weight = kernel
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels * channels, channels, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)

    def __call__(self, x):
        # temp = self.weight.detach().unsqueeze(1).cpu().numpy()
        # for i in range(len(temp)//3):
        #     cv2.imwrite('kernels1/TG/' + str(i) + '.png', temp[i, 0, 0] * 255. * 1)-
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        x = F.conv2d(x, self.weight, padding=0, groups=self.channels)
        return x

class PSFBlurGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, downscale_factor=2, ksize=21, num_kernels=22 ,gaussian_blur_mode=None, bias=False):
        super().__init__()
        self.ksize = ksize
        self.num_kernels = num_kernels
        if not bias:
            self.blur = networks.UnetGenerator(input_nc, output_nc, num_downs, ngf, norm_layer=norm_layer, use_dropout=use_dropout).cuda()
        else:
            self.blur = networks.UnetGenerator(input_nc, output_nc*2, num_downs, ngf, norm_layer=norm_layer, use_dropout=use_dropout).cuda()
        self.gbls = nn.ModuleList([GaussianBlurLayer(mode=gaussian_blur_mode, num_kernels=num_kernels, max_kernel_size=ksize, channels=output_nc).cuda() for _ in range(downscale_factor)])
        self.downscale_factor = downscale_factor
        self.coeffs_learners = nn.ModuleList([networks.ResnetBlurCoefficientGenerator(input_nc, num_kernels).cuda() for _ in range(downscale_factor)])
        self.output_nc = output_nc
        self.input_nc = input_nc
        self.ksize = ksize
        self.num_kernels = num_kernels

    def infer_blur_coeffs(self, blur):
        blurs = []
        for i in range(self.downscale_factor):
            if i == 0:
                b = blur
            else:
                b = F.avg_pool2d(blur, kernel_size=2**i, stride=2**i, padding=0)
            blurs.append(b)
        coeffs = []
        for i in range(self.downscale_factor):
            c = self.coeffs_learners[i](blurs[i])
            coeffs.append(c)
        return torch.stack(coeffs)

    def infer_blurred_clear(self, clear, blur_coeffs):
        blurred_clears = []
        clears = []
        for i in range(self.downscale_factor):
            if i == 0:
                c = clear
            else:
                c = F.avg_pool2d(clear, kernel_size=2**i, stride=2**i, padding=0)
            clears.append(c)
        for i in range(self.downscale_factor):
            clears_blurred = self.gbls[i](clears[i]).view(blur_coeffs.size(1), self.input_nc, self.num_kernels, clears[i].size(2), clears[i].size(3))
            blurred_clears.append((clears_blurred * blur_coeffs[i].unsqueeze(1).unsqueeze(3).unsqueeze(4)).sum(dim=2))
        for i in range(self.downscale_factor):
            blurred_clears[i] = F.interpolate(blurred_clears[i], size=(clear.size(2), clear.size(3)), mode='bilinear', align_corners=False)
        blurred_clear = torch.stack(blurred_clears).mean(dim=0).view(clear.size(0), self.input_nc, clear.size(2), clear.size(3))
        return blurred_clear

    def forward(self, blur, fake_clear, clear, bias_scale=0.05):
        blurs = []
        clears = []
        for i in range(self.downscale_factor):
            if i == 0:
                b = blur
                c = clear
            else:
                b = F.avg_pool2d(blur, kernel_size=2**i, stride=2**i, padding=0)
                c = F.avg_pool2d(clear, kernel_size=2**i, stride=2**i, padding=0)
            blurs.append(b)
            clears.append(c)
        coeffs = []
        for i in range(self.downscale_factor):
            c = self.coeffs_learners[i](blurs[i])
            coeffs.append(c)
        blurred_clears = []
        for i in range(self.downscale_factor):
            clears_blurred = self.gbls[i](clears[i]).view(blur.size(0), self.input_nc, self.num_kernels, clears[i].size(2), clears[i].size(3))
            blurred_clears.append((clears_blurred * coeffs[i].unsqueeze(1).unsqueeze(3).unsqueeze(4)).sum(dim=2))
        for i in range(self.downscale_factor):
            blurred_clears[i] = F.interpolate(blurred_clears[i], size=(blur.size(2), blur.size(3)), mode='bilinear', align_corners=False)
        blurred_clear = torch.stack(blurred_clears).mean(dim=0).view(blur.size(0), self.input_nc, blur.size(2), blur.size(3))
        out = self.blur(blurred_clear)
        out = out[:, :self.output_nc, :, :] #+ bias
        return out, blurred_clear