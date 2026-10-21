
#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J late_fusion
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -o late_fusion%J.out
#BSUB -e late_fusion%J.err

source ~/idlcv_project_2/.venv/bin/activate
cd ~/idlcv_project_2/src/idlcv_project_2

uv run python train.py \
    --data-root /work3/ppar/data/ucf101 \
    --models frame_avg late_fusion early_fusion r3d_18 \
    --epochs 20 \
    --batch-size 2 \
    --accum-steps 4 \
    --num-workers 4 \
    --out-dir runs/late_fusion \
    --test-policy none
