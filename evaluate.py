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

def evaluate(dataroot, data_split_file, model, device, input_nc, crop=True, save=False, save_dir=None, batch_size=1):
    clean_list = []
    noisy_list = []
    with open(os.path.join(dataroot, data_split_file), "r") as f:
        for line in f:
            items = line.strip().split('\t')
            noisy_list.append(items[0])
            clean_list.append(items[1])
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
    ssims = []
    psnrs = []
    lpipss = []
    lpips_loss = lpips.LPIPS(net='vgg').to(device)
    for noisy_file, clean_file in tqdm(zip(noisy_list, clean_list), total=len(noisy_list)):
        if input_nc == 1:
            noisy_img = PIL.Image.open(noisy_file).convert("L")
            clean_img = PIL.Image.open(clean_file).convert("L")
        else:
            noisy_img = PIL.Image.open(noisy_file).convert("RGB")
            clean_img = PIL.Image.open(clean_file).convert("RGB")
        # random crop to 256x256
        if crop:
            w, h = noisy_img.size
            if w < 256 or h < 256:
                continue
            x = np.random.randint(0, w - 256 + 1)
            y = np.random.randint(0, h - 256 + 1)
            noisy_img = noisy_img.crop((x, y, x + 256, y + 256))
            clean_img = clean_img.crop((x, y, x + 256, y + 256))
        else:
            noisy_img = refine_image_size(noisy_img, portion=256)
            clean_img = refine_image_size(clean_img, portion=256)
        model_input = {
            'A': transform(noisy_img).unsqueeze(0).to(device),
            'B': transform(clean_img).unsqueeze(0).to(device),
            'A_paths': noisy_file,
            'B_paths': clean_file}
        output_img = model.netG_A(model_input['A']).detach().cpu().squeeze(0).numpy()
        output_img = output_img.transpose(1, 2, 0)
        output_img = (output_img + 1) / 2
        # plt.imshow(output_img, cmap='gray' if opt.input_nc==1 else None)
        # plt.show()
        if input_nc == 1:
            clean_img = np.expand_dims(clean_img, axis=2)
        if save:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'recon')):
                os.makedirs(os.path.join(save_dir, 'recon'))
            if not os.path.exists(os.path.join(save_dir, 'noisy')):
                os.makedirs(os.path.join(save_dir, 'noisy'))
            if not os.path.exists(os.path.join(save_dir, 'clean')):
                os.makedirs(os.path.join(save_dir, 'clean'))
            output_pil = PIL.Image.fromarray((output_img * 255).astype(np.uint8).squeeze())
            blurred_pil = PIL.Image.fromarray(np.array(noisy_img).astype(np.uint8).squeeze())
            clean_pil = PIL.Image.fromarray(np.array(clean_img).astype(np.uint8).squeeze())
            if crop:
                base_name = os.path.basename(noisy_file).split('.')[0] + f"_crop_{x}_{y}.png"
            else:
                base_name = os.path.basename(noisy_file)
            output_pil.save(os.path.join(save_dir, 'recon', base_name))
            blurred_pil.save(os.path.join(save_dir, 'noisy', base_name))
            clean_pil.save(os.path.join(save_dir, 'clean', base_name))
        ssim_value = ssim(np.array(clean_img)/255, output_img, data_range=1, channel_axis=2)
        psnr_value = psnr(np.array(clean_img)/255, output_img, data_range=1)
        output_img = torch.from_numpy(output_img.transpose(2,0,1)).unsqueeze(0).to(device)
        clean_img = torch.from_numpy(np.array(clean_img, dtype=np.float32).transpose(2,0,1)/255).unsqueeze(0).to(device)
        lpips_value = lpips_loss(
            output_img * 2 - 1,
            clean_img * 2 - 1
        ).item()
        ssims.append(ssim_value)
        psnrs.append(psnr_value)
        lpipss.append(lpips_value)
        # print(f"SSIM: {ssim_value}, PSNR: {psnr_value}, LPIPS: {lpips_value}")
        output_img = None
        clean_img = None
    print(f"Average SSIM: {np.mean(ssims)} std: {np.std(ssims)}, Average PSNR: {np.mean(psnrs)} std: {np.std(psnrs)}, Average LPIPS: {np.mean(lpipss)} std: {np.std(lpipss)}")
    return np.mean(ssims), np.mean(psnrs), np.mean(lpipss)

if __name__ == "__main__":
    np.random.seed(0)
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
    save_dir = os.path.join(opt.checkpoints_dir, opt.name, f"eval_{opt.eval_epoch}")
    print(os.path.join(opt.checkpoints_dir, opt.name, f"eval_{opt.eval_epoch}"))
    evaluate(opt.dataroot, opt.data_split_file, model, opt.device, opt.input_nc, crop=False, save=False, save_dir=save_dir)
