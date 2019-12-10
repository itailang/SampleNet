#!/bin/bash

# train Autoencoder model
python autoencoder/train_ae.py --train_folder log/autoencoder
wait

# evaluate Autoencoder model
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
wait

# train SampleNetProgressive, use Autoencoder model as the task network
python sampler/train_samplenet_progressive.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/SampleNetProgressive
wait

# evaluate SampleNetProgressive
python sampler/evaluate_samplenet_progressive.py --n_sample_points 64 --train_folder log/SampleNetProgressive

# see the results in log/SampleNetProgressive/eval/eval_stats_test_set_multi_0064.txt
