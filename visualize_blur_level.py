"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from options.test_options import TestOptions
import numpy as np
import torch
import PIL
from matplotlib import pyplot as plt
import torchvision
from models.cycle_gan_model import CycleGANModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import lpips

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def refine_image_size(img, portion=256):
    w, h = img.size
    new_w = (w // portion) * portion
    new_h = (h // portion) * portion
    img = img.crop((0, 0, new_w, new_h))
    return img

def visualize_blur_level(blur_img_path, guide_blur_img_path, model, device, input_nc, crop=True):
    if input_nc == 1:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if input_nc == 1:
        blur_img = PIL.Image.open(blur_img_path).convert("L")
        guide_blur_img = PIL.Image.open(guide_blur_img_path).convert("L")
    else:
        blur_img = PIL.Image.open(blur_img_path).convert("RGB")
        guide_blur_img = PIL.Image.open(guide_blur_img_path).convert("RGB")
    if crop:
        w, h = blur_img.size
        if w < 256 or h < 256:
            return
        x = np.random.randint(0, w - 256 + 1)
        y = np.random.randint(0, h - 256 + 1)
        blur_img = blur_img.crop((x, y, x + 256, y + 256))
        guide_blur_img = guide_blur_img.crop((x, y, x + 256, y + 256))
    else:
        blur_img = refine_image_size(blur_img, portion=256)
        guide_blur_img = refine_image_size(guide_blur_img, portion=256)
    model_input = {
        'A': transform(blur_img).unsqueeze(0).to(device),
        'B': transform(guide_blur_img).unsqueeze(0).to(device),
        'A_paths': blur_img_path,
        'B_paths': guide_blur_img_path}
    assert model.netG_B.__class__.__name__ == "PSFBlurGenerator"
    model.netG_B.eval()
    model.set_input(model_input)
    model.forward()
    coeffs = model.netG_B.infer_blur_coeffs(model.real_B)
    model.blurred_clear_fake_B = model.netG_B.infer_blurred_clear(model.fake_B, coeffs)
    model.blurred_clear_real_B = model.netG_B.infer_blurred_clear(model.real_B, coeffs)

    model.rec_A = model.netG_B.blur(model.blurred_clear_fake_B)
    blurred_output = model.netG_B.blur(model.blurred_clear_real_B)
    # _, blurred_output = model.netG_B(model_input['A'], None, model_input['B'])
    blurred_output = blurred_output.squeeze(0).cpu().detach().numpy()
    blurred_output = blurred_output.transpose(1, 2, 0)
    blurred_output = (blurred_output + 1) / 2
    model.blurred_clear_fake_B = model.blurred_clear_fake_B.squeeze(0).cpu().detach().numpy()
    model.blurred_clear_fake_B = model.blurred_clear_fake_B.transpose(1, 2, 0)
    model.blurred_clear_fake_B = (model.blurred_clear_fake_B + 1) / 2
    # model.rec_B = model.rec_B.squeeze(0).cpu().detach().numpy()
    # model.rec_B = model.rec_B.transpose(1, 2, 0)
    # model.rec_B = (model.rec_B + 1) / 2
    model.rec_A = model.rec_A.squeeze(0).cpu().detach().numpy()
    model.rec_A = model.rec_A.transpose(1, 2, 0)
    model.rec_A = (model.rec_A + 1) / 2
    model.fake_B = model.fake_B.squeeze(0).cpu().detach().numpy()
    model.fake_B = model.fake_B.transpose(1, 2, 0)
    model.fake_B = (model.fake_B + 1) / 2
    # plt.subplot(1,3,1)
    # plt.title("FFT of Generated Blurry Image")
    # f = np.fft.fft2(blurred_output)
    # magnitude_spectrum = 20 * np.log(np.abs(f))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1,3,2)
    # plt.title("FFT of Guidance Blurry Image")
    # f = np.fft.fft2(np.array(blur_img)/255.0)
    # magnitude_spectrum = 20 * np.log(np.abs(f))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.axis('off')
    # plt.subplot(1,3,3)
    # plt.title("FFT of Clear Image")
    # f = np.fft.fft2(np.array(clear_img)/255.0)
    # magnitude_spectrum = 20 * np.log(np.abs(f))
    # plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.axis('off')
    # plt.show()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Guidance Blurry Image")
    plt.imshow(np.array(guide_blur_img), cmap='gray' if input_nc==1 else None)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Clear Image")
    plt.imshow(np.array(model.fake_B), cmap='gray' if input_nc==1 else None)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Generated Blurry Image")
    plt.imshow(model.rec_A, cmap='gray' if input_nc==1 else None)
    plt.axis('off')
    plt.show()

blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_16/16_a12_s2_w1.png'
guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_02/02_a12_s1_w1.png'
# guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_05/05_a12_s1_w1.png'
# guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_09/09_a12_s1_w1.png'
# guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_23/23_a12_s1_w1.png'
# guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_30/30_a12_s1_w1.png'
# guide_blur_img_path = '/home/sixuan-wu/datasets/bbbc006/BBBC006_v1_images_z_16/16_a12_s1_w1.png'

if __name__ == "__main__":
    np.random.seed(2)
    opt = TestOptions().parse()  # get test options
    print(opt.model)
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    
    model = CycleGANModel(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.load_networks(opt.eval_epoch)
    model.eval()
    visualize_blur_level(blur_img_path, guide_blur_img_path, model, opt.device, opt.input_nc, crop=True)
