#!/bin/bash
# Ablation studies for CS 445 project
set -e

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting
export CUDA_HOME=$CONDA_PREFIX

SCENE="data/nerf_synthetic/lego"
BASE_ARGS="--eval --white_background --quiet"

echo "=========================================="
echo "Ablation 1: SH Degree (0, 1, 2, 3)"
echo "=========================================="
for SH in 0 1 2 3; do
    OUT="output/ablation_sh${SH}"
    echo "Training SH degree=${SH}..."
    python train.py -s $SCENE -m $OUT $BASE_ARGS \
        --sh_degree $SH --iterations 30000 \
        --save_iterations 30000 --test_iterations 30000
    python render.py -m $OUT --skip_train --quiet
    python metrics.py -m $OUT 2>&1 | grep -E "SSIM|PSNR|LPIPS"
    echo "---"
done

echo "=========================================="
echo "Ablation 2: Training Iterations (convergence)"
echo "=========================================="
# Train once to 30k, saving checkpoints at each milestone
OUT="output/ablation_convergence"
echo "Training to 30k with checkpoints..."
python train.py -s $SCENE -m $OUT $BASE_ARGS \
    --iterations 30000 \
    --save_iterations 1000 3000 5000 7000 10000 15000 20000 30000 \
    --test_iterations 1000 3000 5000 7000 10000 15000 20000 30000

for ITER in 1000 3000 5000 7000 10000 15000 20000 30000; do
    echo "Rendering iteration ${ITER}..."
    python render.py -m $OUT --iteration $ITER --skip_train --quiet
    python metrics.py -m $OUT 2>&1 | grep -E "Method|SSIM|PSNR|LPIPS"
    echo "---"
done

echo "=========================================="
echo "Ablation 3: Densification Threshold"
echo "=========================================="
for THRESH in 0.0001 0.0002 0.0004 0.0008; do
    OUT="output/ablation_densify_${THRESH}"
    echo "Training densify_grad_threshold=${THRESH}..."
    python train.py -s $SCENE -m $OUT $BASE_ARGS \
        --densify_grad_threshold $THRESH --iterations 30000 \
        --save_iterations 30000 --test_iterations 30000
    python render.py -m $OUT --skip_train --quiet
    python metrics.py -m $OUT 2>&1 | grep -E "SSIM|PSNR|LPIPS"
    # Count gaussians
    python -c "from plyfile import PlyData; p=PlyData.read('${OUT}/point_cloud/iteration_30000/point_cloud.ply'); print(f'  Gaussians: {p[\"vertex\"].count:,}')"
    echo "---"
done

echo "All ablations complete!"
