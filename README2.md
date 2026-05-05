# µClearGAN: A Blurriness Level Guided GAN for Unsupervised Microscopic Image Deblurring

> Accepted at the **48th Annual IEEE EMBC 2026** · Toronto, Canada  
> Sixuan Wu · Anika Gaddam · Alexander Adams · Uncommon Sense Labs

---

## Overview

µClearGAN is an end-to-end PyTorch framework for **unsupervised microscopic image deblurring**. It combines:

- A **ResNet18 blur-intensity regressor** with aleatoric uncertainty estimation to predict blur kernel size from image patches.
- A **CycleGAN-based deblurring model** with dual discriminators and a learned blur-kernel generator to translate blurry microscopy images into sharp ones — without paired training data.

The framework supports grayscale and RGB microscopy images and has been validated on the **BBBC006**, **RBC**, and **histology** datasets.

---

## Requirements

- Windows, Linux, or macOS
- Python 3.11
- PyTorch 2.4+ with CUDA 12.1 (CPU also supported)
- conda (recommended)

> **Note:** The training/testing commands in this README use bash syntax. On Windows, run them in **Git Bash** — they will not work in PowerShell or Command Prompt.

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/wsxwsx543/MicroClearGAN.git
cd MicroClearGAN
```

**2. Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

**3. Install additional dependencies**

```bash
pip install matplotlib focal-frequency-loss pytorch-msssim omegaconf lpips scikit-image wandb timm einops scipy fvcore
```

---

## Dataset Preparation

### Downloading the Datasets

The datasets are not included in this repository and must be downloaded separately.

| Dataset | Download Link | Channels |
|---|---|---|
| BBBC006 (W1 & W2) | [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC006) | Grayscale |
| FocusPath | [INSERT LINK HERE] | RGB |
| RBC (Ours) | [INSERT LINK HERE] | RGB |
| Histology | [INSERT LINK HERE] | RGB |

### Setting Up Split Files

The model uses a **split file** (`.txt`) to pair blurry (domain A) and sharp (domain B) image paths. Each line is tab-separated:
/path/to/blurry/image.png	/path/to/sharp/image.png

> **Important:** After downloading the datasets, update the image paths inside each split file to match the actual location of your images on your machine. For example, replace `/path/to/blurry/image.png` with the real path such as `/home/user/datasets/BBBC006/blurry/img001.png`.

Pre-made split files are provided for each dataset:

| Split File | Dataset | Channels |
|---|---|---|
| `train_bbbc006_w1.txt` | BBBC006 (W1) | Grayscale |
| `train_bbbc006_w2.txt` | BBBC006 (W2) | Grayscale |
| `train_rbc.txt` | RBC | RGB |
| `train_hist.txt` | Histology | RGB |
| `train_focus.txt` | Focus dataset | RGB |

Place your split files and image data under a `data_splits/` directory at the repo root:
data_splits/
├── train_bbbc006_w1.txt
├── train_bbbc006_w2.txt
├── train_rbc.txt
├── train_hist.txt
├── train_focus.txt
└── [your images referenced in the split files]

---

## Training

### Grayscale datasets (BBBC006)

```bash
python train.py \
  --dataroot ./data_splits \
  --checkpoints_dir ./ckpt \
  --data_split_file train_bbbc006_w1.txt \
  --name bbbc006_w1 \
  --model cycle_gan \
  --n_epochs 15 \
  --n_epochs_decay 15 \
  --input_nc 1 \
  --output_nc 1
```

### RGB datasets (RBC, Histology)

```bash
# RBC
python train.py \
  --dataroot ./data_splits \
  --checkpoints_dir ./ckpt \
  --data_split_file train_rbc.txt \
  --name rbc \
  --model cycle_gan \
  --n_epochs 10 \
  --n_epochs_decay 10

# Histology
python train.py \
  --dataroot ./data_splits \
  --checkpoints_dir ./ckpt \
  --data_split_file train_hist.txt \
  --name hist \
  --model cycle_gan \
  --n_epochs 10 \
  --n_epochs_decay 10
```

### W&B experiment tracking

Add `--use_wandb` to any training command to log losses and sample images to your W&B dashboard:

```bash
python train.py ... --use_wandb
```

### Multi-GPU training

```bash
torchrun --nproc_per_node=4 train.py ... --norm sync_batch
```

> Note: `--norm batch` is not compatible with DDP. Use `--norm sync_batch` or `--norm sync_instance`.

Checkpoints are saved to `./ckpt/{name}/` by default. Training progress (sample images) can be viewed at `./ckpt/{name}/web/index.html`.

---

## Testing / Inference

> **Note:** You must complete training before running testing. The test script loads the trained checkpoint from `./ckpt/{name}/latest_net_G_A.pth`.

```bash
python test.py \
  --dataroot ./data_splits \
  --checkpoints_dir ./ckpt \
  --data_split_file train_bbbc006_w1.txt \
  --name bbbc006_w1 \
  --model cycle_gan \
  --input_nc 1 \
  --output_nc 1
```

Results are saved to `./results/{name}/latest_test/index.html`.

To run inference on a single directory of blurry images (one-sided translation):

```bash
python test.py \
  --dataroot ./path/to/blurry_images \
  --checkpoints_dir ./ckpt \
  --name bbbc006_w1 \
  --model test \
  --no_dropout \
  --input_nc 1
```

---

## Evaluation

> **Note:** Run testing first to generate results before running evaluation.

```bash
python evaluate.py \
  --results_dir ./results/bbbc006_w1/latest_test \
  --checkpoints_dir ./ckpt \
  --name bbbc006_w1
```

---

## Repository Structure

    MicroClearGAN/
    ├── data/
    │   ├── micro_unaligned_dataset.py   # Custom dataset loader using split files
    │   └── base_dataset.py
    ├── models/
    │   ├── cycle_gan_model.py           # CycleGAN training logic
    │   ├── networks.py                  # Generator/discriminator architectures
    │   ├── loss.py                      # Custom loss functions
    │   ├── psf_blur/                    # Blur kernel (PSF) simulation
    │   └── grl/                        # GRL attention blocks
    ├── options/
    │   ├── base_options.py
    │   ├── train_options.py
    │   └── test_options.py
    ├── datasets/                        # Upstream CycleGAN dataset scripts (not used for microscopy datasets)
    ├── docs/                            # Tips, FAQ, Docker guide
    ├── run.sh                           # Example training commands
    ├── run_experiment.sh                # Full experiment sweep
    ├── environment.yml                  # Conda environment
    ├── train.py
    ├── test.py
    └── evaluate.py

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{wu2026micleargan,
  title={µClearGAN: A Blurriness Level Guided GAN for Unsupervised Microscopic Image Deblurring},
  author={Wu, Sixuan and Gaddam, Anika and Adams, Alexander},
  booktitle={48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2026}
}
```

---

## Acknowledgments

This project builds on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu and Taesung Park. Research conducted at Uncommon Sense Labs.

---

## Acknowledgments

This project builds on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by Jun-Yan Zhu and Taesung Park. Research conducted at Uncommon Sense Labs.
