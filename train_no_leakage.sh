
#!/bin/sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J no_leakage
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -o no_leakage%J.out
#BSUB -e no_leakage%J.err

source ~/idlcv_project_2/.venv/bin/activate
cd ~/idlcv_project_2/src/idlcv_project_2

uv run python train.py \
    --data-root /dtu/datasets1/02516/ucf101_noleakage \
    --models late_fusion \
    --epochs 20 \
    --pretrained \
    --batch-size 16 \
    --accum-steps 4 \
    --num-workers 4 \
    --out-dir runs/no_leakage \
    --test-policy none
