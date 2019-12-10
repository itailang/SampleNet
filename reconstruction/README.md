## Installation

The code has been tested with Python 3.6.9, TensorFlow 1.13.2, TFLearn 0.3.2, CUDA 10.0 and cuDNN 7.6.2 on Ubuntu 16.04.

Install <a href="https://www.tensorflow.org/install" target="_blank">TensorFlow</a>. You can also use a <a href="https://www.tensorflow.org/install/docker" target="_blank">TensorFlow Docker image</a>. A <a href="https://hub.docker.com/r/tensorflow/tensorflow/" target="_blank">Docker image</a> that meets the TensorFlow, CUDA and cuDNN version that we used is **tensorflow/tensorflow:1.13.2-gpu-py3**.

Install <a href="http://tflearn.org/installation" target="_blank">TFLearn</a>.

Install <a href="https://matplotlib.org/users/installing.html" target="_blank">matplotlib</a> (for visualization of point clouds).

In order to download the dataset for training and evaluation, wget package is required. To install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

### Compilation of TensorFlow ops 

Compile TensorFlow ops: nearest neighbor grouping and farthest point sampling, implemented by [Qi et al.](https://github.com/charlesq34/pointnet2); structural losses, implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration) The ops are located under `reconstruction/external` at `grouping`, `sampling`, and `structural losses` folders, respectively. If needed, use a text editor and modify the corresponding `sh` file of each op to point to your `nvcc` path. Then, use:   
```bash
cd reconstruction/
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

These scripts train and evaluate an Autoencoder model with complete point clouds, use it to train a sampler (SampleNet or SampleNetProgressive), and then evaluate sampler by running its sampled points through the Autoencoder model. In the following sections, we explain how to run each part of this pipeline separately. 

### Data Set

Point clouds of <a href="https://www.shapenet.org/" target="_blank">ShapeNetCore</a> models in `ply` files (provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a>) will be automatically downloaded (1.4GB) on the first training of an Autoencoder model. Each point cloud contains 2048 points, uniformly sampled from a shape surface. The data will be downloaded to the folder `reconstruction/data/shape_net_core_uniform_samples_2048`.    

### Autoencoder

To train an Autoencoder model, use:
```bash
python autoencoder/train_ae.py --train_folder log/autoencoder
```

To evaluate an Autoencoder model, use:
```bash
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
```

This evaluation script saves the reconstructed point clouds of the test set, and the reconstruction error per point cloud (Chamfer distance between the input and reconstruction).  The results are saved to the `train_folder`.

### SampleNet
To train SampleNet (with sample size 64 in this example), using an existing Autoencoder model as the task network (provided in `ae_folder` argument), use:
```bash
python sampler/train_samplenet.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/SampleNet64
```

To evaluate reconstruction with SampleNet's sampled points (with sample size 64 in this example), use:
```bash
python sampler/evaluate_samplenet.py --train_folder log/SampleNet64
```

This evaluation script saves the sampled point clouds, sample indices and reconstructed point clouds of the test set, and the reconstruction error per point cloud (Chamfer distance between the input and reconstruction). It also computes the normalized reconstruction error, as explained in the paper. The results are saved to the `train_folder`.

### SampleNetProgressive
To train SampleNetProgressive, using an existing Autoencoder model as the task network (provided in `ae_folder` argument), use:
```bash
python sampler/train_samplenet_progressive.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/SampleNetProgressive
```

To evaluate reconstruction with SampleNetProgressive's sampled points (with sample size 64 in this example), use:
```bash
python sampler/evaluate_samplenet_progressive.py --n_sample_points 64 --train_folder log/SampleNetProgressive
```

This script operates similarly to the evaluation script for SampleNet. 

### Visualization
You can visualized point clouds (input, reconstructed, or sampled) by adding the flag `--visualize_results` to the evaluation script of the Autoencoder, SampleNet or SampleNetProgressive.  

## Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a> and <a href="https://github.com/orendv/learning_to_sample" target="_blank">Dovrat et al.</a> We thank the authors for sharing their code.
