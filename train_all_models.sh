
#!/bin/sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J all_models
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -o all_models%J.out
#BSUB -e all_models%J.err

source ~/idlcv_project_2/.venv/bin/activate
cd ~/idlcv_project_2/src/idlcv_project_2

uv run python train.py \
    --data-root /dtu/datasets1/02516/ufc10 \
    --models frame_avg late_fusion early_fusion r3d_18 \
    --epochs 20 \
    --batch-size 2 \
    --accum-steps 4 \
    --num-workers 4 \
    --out-dir runs/part41 \
    --test-policy none
