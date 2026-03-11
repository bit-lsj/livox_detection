## Features
- Anchor-free method
- Support multiple point cloud datasets with different patterns.

> Note: This repo is modified from [Livox-SDK/livox_detection](https://github.com/Livox-SDK/livox_detection) to support **CPU-only** inference (no CUDA build required).

## Setup
This repo supports **GPU (CUDA)** and **CPU-only** inference.

### 1) Get the code

```bash
# Clone (or skip if you already have it)
cd ~
git clone https://github.com/bit-lsj/livox_detection.git
cd ~/livox_detection
```

### 2) System dependencies (Conda + ROS tools)

#### 2.1) Install Anaconda (recommended)

```bash
cd ~
wget -c https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
bash Anaconda3-2025.12-2-Linux-x86_64.sh

# Make conda available in new shells
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
```

#### 2.2) Install ROS runtime deps

```bash
sudo apt update

# ROS (melodic/noetic are both common; pick the one you actually use)
# If you already have ROS installed, you can skip ROS installation steps and only install the packages below.

sudo apt install -y python3-dev
sudo apt install -y ros-noetic-rospy ros-noetic-sensor-msgs ros-noetic-geometry-msgs
sudo apt install -y ros-noetic-ros-numpy ros-noetic-rviz ros-noetic-rosbag
```

#### 2.3) Fix ros_numpy with newer NumPy (ROS Noetic common issue)

If you see `AttributeError: module 'numpy' has no attribute 'float'` when importing `ros_numpy`, patch it once:

```bash
sudo sed -i 's/dtype=np\.float)/dtype=np.float64)/' /opt/ros/noetic/lib/python3/dist-packages/ros_numpy/point_cloud2.py
```

### 3) Python environment

#### Option A: GPU (CUDA) environment (recommended for real-time)

```bash
# conda is recommended (python 3.8)
conda create -n livox_det python=3.8 -y
conda activate livox_det

# Check max CUDA version supported by current NVIDIA driver
nvidia-smi

# PyTorch GPU
# CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyyaml rospkg
```

Build/install this project (builds CUDA extension for NMS):

```bash
python3 setup.py develop
```

#### Option B: CPU-only environment (no CUDA / no compilation)

```bash
conda create -n livox_det_cpu python=3.8 -y
conda activate livox_det_cpu

# CPU-only PyTorch
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyyaml rospkg

# IMPORTANT: do NOT run setup.py on CPU-only machines (it tries to compile CUDA extension)
# Instead, run scripts directly by adding repo root to PYTHONPATH (run this under livox_detection/):
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Usage
1. Run ROS.
```
roscore
```
2. Move to 'tools' directory and run test_ros.py (pretrained model: ../pt/livox_model_1.pt or ../pt/livox_model_2.pt).
```
cd ~/livox_detection/tools
python3 test_ros.py --pt ../pt/livox_model_1.pt
```
3. Play rosbag. (Please adjust the ground plane to 0m and keep it horizontal. The topic of pointcloud2 should be /livox/lidar)
```
rosbag play bags/highwayscene1.bag
```
4. Visualize the results.
```
rviz -d rviz.rviz
```
