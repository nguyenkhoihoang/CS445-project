# Fixing Background Artifacts in 3D Gaussian Splatting

**CS 445 Final Project** — Paper Reproduction + Bug Fix + Ablation Studies

> Based on [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl et al. (SIGGRAPH 2023)

---

## Our Contribution

We identified and fixed a **training bug** in the official 3DGS implementation that causes severe background floater artifacts on any dataset with alpha masks (e.g., NeRF Synthetic/Blender scenes).

### The Bug

In [`train.py`](train.py), the original code masks the rendered image to zero in transparent regions before computing the loss:

```python
# ORIGINAL (buggy):
if viewpoint_cam.alpha_mask is not None:
    image *= alpha_mask          # render background → zero
gt_image = viewpoint_cam.original_image  # GT background is already zero
loss = L1(image, gt_image)       # loss in background = |0 - 0| = 0 always!
```

**Problem**: Background loss is always zero regardless of what Gaussians exist there. The optimizer receives no gradient signal to remove spurious Gaussians in transparent regions, so they accumulate as floater artifacts.

### Our Fix

Instead of masking the render, we composite the ground truth with the same background color:

```python
# OUR FIX:
gt_image = viewpoint_cam.original_image
if viewpoint_cam.alpha_mask is not None:
    gt_image = gt_image * alpha_mask + bg[:, None, None] * (1 - alpha_mask)
loss = L1(image, gt_image)       # floaters now penalized!
```

### Results

| | Original (bug) | Our Fix | Change |
|:---|:---:|:---:|:---:|
| **PSNR** | 14.27 dB | **35.94 dB** | +21.67 |
| **SSIM** | 0.773 | **0.983** | +0.210 |
| **LPIPS** | 0.307 | **0.015** | -0.292 |
| Gaussians | 239,886 | 297,788 | +24% |
| Render Speed | 3.7 it/s | 5.8 it/s | +57% |


The fixed model uses 24% more Gaussians yet renders 57% faster. The buggy model's floater Gaussians are large and overlapping, causing expensive per-pixel compositing. The fixed model allocates smaller, well-placed Gaussians to actual scene content.

---

## Quick Start

### Setup

```bash
# Create environment
conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting

# Install PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CUDA toolkit for compiling extensions
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc=11.8 cuda-cudart-dev -y
cp -rn $CONDA_PREFIX/targets/x86_64-linux/include/* $CONDA_PREFIX/include/ 2>/dev/null
conda remove gxx_linux-64 gcc_linux-64 gxx_impl_linux-64 gcc_impl_linux-64 -y 2>/dev/null

# Install dependencies
pip install plyfile tqdm opencv-python joblib lpips
export CUDA_HOME=$CONDA_PREFIX
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```

### Train

```bash
# NeRF Synthetic (with our fix)
python train.py -s data/nerf_synthetic/lego -m output/lego --eval --white_background

# Your own images (after COLMAP processing)
python convert.py -s data/my_scene
python train.py -s data/my_scene -m output/my_scene
```

### Render & Evaluate

```bash
python render.py -m output/lego
python metrics.py -m output/lego
```

---

## Project Files

| File | Description |
|:---|:---|
| [`train.py`](train.py) | Training script (**with our fix**) |
| [`render.py`](render.py) | Rendering script (with GT compositing fix) |
| [`run_ablations.sh`](run_ablations.sh) | Ablation study automation script |
| [`output/figures/`](output/figures/) | All generated figures |

Real data folder: [Drive Link](https://drive.google.com/drive/folders/10qwBJsxlxle76cUn8wSYVPgIsfOYbLjv?usp=drive_link)

---

## Acknowledgments

This project builds upon the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . Our contributions (the bug fix, ablation studies, and analysis) are built on top of their work.



**Original paper**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (ACM Transactions on Graphics, SIGGRAPH 2023)


