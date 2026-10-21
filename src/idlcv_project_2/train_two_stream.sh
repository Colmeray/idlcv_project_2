#!/bin/sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J no_leakage
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -o no_leakage%J.out
#BSUB -e no_leakage%J.err

uv run python train.py \
    --data-root /dtu/datasets1/02516/ucf101_noleakage \
    --models two_stream \
    --flow-root /dtu/datasets1/02516/ucf101_noleakage/flows-png \
    --epochs 20 \
    --batch-size 2 \
    --accum-steps 4 \
    --num-workers 4 \
    --out-dir runs/two_stream \
    --test-policy none
