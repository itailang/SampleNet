#!/bin/bash

# train Autoencoder model
python autoencoder/train_ae.py --train_folder log/autoencoder
wait

# evaluate Autoencoder model
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
wait

# train SampleNet, use Autoencoder model as the task network
python sampler/train_samplenet.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/SampleNet64
wait

# evaluate SampleNet
python sampler/evaluate_samplenet.py --train_folder log/SampleNet64

# see the results in log/SampleNet64/eval/eval_stats_test_set_multi_0064.txt
