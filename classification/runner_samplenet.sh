#!/bin/bash

# train PointNet classifier
python train_classifier.py --model pointnet_cls --log_dir log/PointNet1024
wait

# train SampleNet, use PointNet classifier as the task network
python train_samplenet.py --classifier_model pointnet_cls --classifier_model_path log/PointNet1024/model.ckpt \
    --num_out_points 32 --log_dir log/SampleNet32
wait

# infer SampleNet and evaluate PointNet classifier with sampled points of SampleNet
python evaluate_samplenet.py --sampler_model_path log/SampleNet32/model.ckpt \
    --num_out_points 32 --dump_dir log/SampleNet32/eval

# see the results in log/SampleNet32/eval/log_evaluate.txt
