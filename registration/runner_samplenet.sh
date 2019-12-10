#!/bin/bash

# Train PCR-Net
python main.py \
    -o log/baseline/PCRNet1024 \
    --datafolder car_hdf5_2048 \
    --sampler none \
    --train-pcrnet \
    --epochs 500 \

wait

# Train SampleNet
python main.py \
    -o log/SAMPLENET64 \
    --datafolder car_hdf5_2048 \
    --transfer-from log/baseline/PCRNet1024_model_best.pth \
    --sampler samplenet \
    --train-samplenet \
    --num-out-points 64 \
    --epochs 400 \

wait
