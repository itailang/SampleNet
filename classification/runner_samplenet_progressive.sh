#!/bin/bash

# train PointNet (vanilla) classifier
python train_classifier.py --model pointnet_cls_basic --log_dir log/PointNetVanilla1024
wait

# train SampleNetProgressive, use PointNet (vanilla) classifier as the task network
python train_samplenet_progressive.py --classifier_model pointnet_cls_basic --classifier_model_path log/PointNetVanilla1024/model.ckpt \
    --log_dir log/SampleNetProgressive
wait

# infer SampleNetProgressive and save the ordered point clouds to .h5 files
python infer_samplenet_progressive.py --sampler_model_path log/SampleNetProgressive/model.ckpt \
    --dump_dir log/SampleNetProgressive
wait

# evaluate PointNet (vanilla) classifier with sampled points of SampleNetProgressive
python evaluate_from_files.py --classifier_model pointnet_cls_basic --classifier_model_path log/PointNetVanilla1024/model.ckpt \
    --data_path log/SampleNetProgressive/sampled --dump_dir log/SampleNetProgressive/eval/sampled

# see the results in log/SampleNetProgressive/eval/sampled/log_evaluate.txt
