## Installation

The code has been tested with Python 3.6.9, TensorFlow 1.13.2, CUDA 10.0 and cuDNN 7.6.2 on Ubuntu 16.04.

Install <a href="https://www.tensorflow.org/install" target="_blank">TensorFlow</a>. You can also use a <a href="https://www.tensorflow.org/install/docker" target="_blank">TensorFlow Docker image</a>. A <a href="https://hub.docker.com/r/tensorflow/tensorflow/" target="_blank">Docker image</a> that meets the TensorFlow, CUDA and cuDNN version that we used is **tensorflow/tensorflow:1.13.2-gpu-py3**.

Install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

In order to download the dataset for training and evaluation, wget package is required. To install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

### Compilation of TensorFlow ops 

Compile TensorFlow ops: nearest neighbor grouping, implemented by [Qi et al.](https://github.com/charlesq34/pointnet2); structural losses, implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration) The ops are located under `classification` at `grouping` and `structural losses` folders, respectively. If needed, use a text editor and modify the corresponding `sh` file of each op to point to your `nvcc` path. Then, use:   
```bash
cd classification/
sh compile_ops.sh
```

An `o` and `so` files should be created in the corresponding folder of each op. 

## Usage
For a quick start please use:
```bash
sh runner_samplenet.sh
 ```
or:
```bash
sh runner_samplenet_progressive.sh
```

These scripts train a classifier model with complete point clouds, use it to train a sampler (SampleNet or SampleNetProgressive), and then evaluate sampler by running its sampled points through the classifier model. In the following sections, we explain how to run each part of this pipeline separately.

### Data Set

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in `h5` files (provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a>) will be automatically downloaded (416MB) on the first training of a classifier model. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also `json` files specifying the IDs of shapes in the `h5` files. The data will be downloaded to the folder `classification/data/modelnet40_ply_hdf5_2048`. 

### Classifier

To train a PointNet model to classify point clouds, use:
```bash
python train_classifier.py --model pointnet_cls --log_dir log/PointNet1024
```

To train the vanilla version of PointNet, use:
```bash
python train_classifier.py --model pointnet_cls_basic --log_dir log/PointNetVanilla1024
```

### SampleNet
To train SampleNet (with sample size 32 in this example), using an existing classifier (PointNet in this example) as the task network (provided in `classifier_model_path` argument), use:
```bash
python train_samplenet.py --classifier_model pointnet_cls --classifier_model_path log/PointNet1024/model.ckpt --num_out_points 32 --log_dir log/SampleNet32
```

To evaluate classification with SampleNet's sampled points (with sample size 32 in this example), use:
```bash
python evaluate_samplenet.py --sampler_model_path log/SampleNet32/model.ckpt --num_out_points 32 --dump_dir log/SampleNet32/eval
```

This evaluation script computes classification accuracy results and saves them to the `dump_dir`.  

### SampleNetProgressive
To train SampleNetProgressive, using an existing classifier (PointNet vanilla in this example) as the task network  (provided in `classifier_model_path` argument), use:
```bash
python train_samplenet_progressive.py --classifier_model pointnet_cls_basic --classifier_model_path log/PointNetVanilla1024/model.ckpt --log_dir log/SampleNetProgressive
```

Evaluation of SampleNetProgressive is done in two steps. First, infer SampleNetProgressive and save the ordered point clouds to `h5` files:
```bash
python infer_samplenet_progressive.py --sampler_model_path log/SampleNetProgressive/model.ckpt --dump_dir log/SampleNetProgressive
```

Then, evaluate the classifier using SampleNetProgressive's sampled points:
```bash
python evaluate_from_files.py --classifier_model pointnet_cls_basic --classifier_model_path log/PointNetVanilla1024/model.ckpt --data_path log/SampleNetProgressive/sampled --dump_dir log/SampleNetProgressive/eval/sampled
```

This evaluation script computes classification accuracy results for different sample sizes and saves them to the `dump_dir`.

## Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a> and <a href="https://github.com/orendv/learning_to_sample" target="_blank">Dovrat et al.</a> We thank the authors for sharing their code.
