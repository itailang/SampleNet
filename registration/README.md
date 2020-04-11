## Installation
We strongly recommend working with <a href="https://hub.docker.com/search/?type=edition&offering=community" target="_blank">Docker Engine</a> and <a href="https://github.com/NVIDIA/nvidia-docker/tree/master">Nvidia-Docker</a>.
At this moment, the container can only run on a CUDA (_linux/amd64_) enabled machine due to specific compiled ops from <a href="https://github.com/erikwijmans/Pointnet2_PyTorch">Pointnet2_PyTorch</a>.

### Pull and run the Docker container
```bash
docker pull asafmanor/pytorch:samplenetreg_torch1.4
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name samplenetreg asafmanor/pytorch:samplenetreg_torch1.4
```

### Alternatively, build your own Docker image
#### On the host machine
```bash
docker build -t samplenetreg_torch1.4_image .
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name samplenetreg samplenetreg_torch1.4_image
```
#### Inside the Docker container
```bash
cd /root/Pointnet2_PyTorch
git checkout 5ff4382f56a8cbed2b5edd3572f97436271aba89
pip install -r requirements.txt
pip install -e .
cd /workspace
```

## Usage
### Data preparation
Create the 'car' dataset (ModelNet40 data will automatically be downloaded to `data/modelnet40_ply_hdf5_2048` if needed) and log directories:
```bash
mkdir log
mkdir log/baseline
python data/create_dataset_torch.py
```
Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files (provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a>) will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

### Training and evaluating
For a quick start please use:
```bash
sh runner_samplenet.sh
```

### Train *PCRNet* (supervised) registration network
To train a *PCRNet* model to register point clouds, use:
```bash
python main.py -o log/baseline/PCRNet1024 --datafolder car_hdf5_2048 --sampler none --train-pcrnet --epochs 500
```

### Train SampleNet
To train SampleNet (with sample size 64 in this example), using an existing PCRNet as the task network, use:
```bash
python main.py -o log/SAMPLENET64 --datafolder car_hdf5_2048 --transfer-from log/baseline/PCRNet1024_model_best.pth --sampler samplenet --train-samplenet --num-out-points 64
```

### Evaluate SampleNet
To evaluate PCRNet with SampleNet's sampled points (with sample size 64 in this example), use:
```bash
python main.py -o log/SAMPLENET64  --datafolder car_hdf5_2048 --pretrained log/SAMPLENET64_model_best.pth --sampler samplenet --num-out-points 64 --test
```

Additional options for training and evaluating can be found using `python main.py --help`.

## Acknowledgment
This code builds upon the code provided in <a href="https://github.com/hmgoforth/PointNetLK">PointNetLK</a>, <a href="https://github.com/erikwijmans/Pointnet2_PyTorch">Pointnet2_PyTorch</a> and <a href="https://github.com/unlimblue/KNN_CUDA">KNN_CUDA</a>. We thank the authors for sharing their code.

