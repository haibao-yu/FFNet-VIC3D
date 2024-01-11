![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# FFNET: Flow-Based Feature Fusion for VEHICLE-INFRASTRUCTURE COOPERATIVE 3D OBJECT DETECTION

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/FFNET-OVERVIEW.png" height="400">
</div>
<p align="center">
  Figure 1: FFNET OVERVIEW.
</p>

### [Project page](https://github.com/haibao-yu/FFNet-VIC3D) | [Paper](https://arxiv.org/abs/2311.01682) |

FFNET: Flow-Based Feature Fusion for VEHICLE-INFRASTRUCTURE COOPERATIVE 3D OBJECT DETECTION.<br>
[Haibao Yu](https://scholar.google.com/citations?user=JW4F5HoAAAAJ), Yingjuan Tang, [Enze Xie](https://xieenze.github.io/), Jilei Mao, [Ping Luo](http://luoping.me/), and [Zaiqing Nie](https://air.tsinghua.edu.cn/en/info/1046/1192.htm) <br>
NeurIPS 2023.

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for [FFNET](https://openreview.net/forum?id=ZLfD0cowleE).

FFNET is a simple, efficient and powerful VIC3D Object Detection method, as shown in Figure 1.

We use [MMDetection3D v0.17.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1) as the codebase. <br>
We evaluation all the models with [OpenDAIRV2X](https://github.com/AIR-THU/DAIR-V2X).


## Installation
For more information about installing mmdet3d, please refer to the guidelines in [MMDetectionn3D v0.17.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1).
For more information about installing OpenDAIRV2X, please refer to the guideline in [OpenDAIRV2X](https://github.com/AIR-THU/DAIR-V2X).


Other requirements:
```pip install --upgrade git+https://github.com/klintan/pypcd.git```

An example (works for me): ```CUDA 11.1``` and  ```pytorch 1.9.0``` 

```
pip install torchvision==0.10.0
pip install mmcv-full==1.3.14
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
cd FFNET-VIC3D && pip install -e . --user
```

## Data Preparation
We train and evaluate the models on DAIR-V2X dataset. For downloading DAIR-V2X dataset, please refer to the guidelines in [DAIR-V2X](https://thudair.baai.ac.cn/cooptest).
After downloading the dataset, we should preprcocess the dataset as the guidelines in [data_preprocess](data/dair-v2x/README.md).
We provide the preprocessed example data [example-cooperative-vehicle-infrastructure](https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=sharing), you can download and decompress it under './data/dair-v2x'.


## Evaluation

Download `trained weights`. 
(
[FFNET Trainded Checkpoint](https://drive.google.com/file/d/1eX2wZ7vSxq8y9lAyjHyrmBQ30qNHcFC6/view?usp=sharing) | [FFNET without prediction](https://drive.google.com/file/d/14ujtkGVMGGdvHnmEAUDArny6HKbYM_ye/view?usp=sharing) 
| [FFNET-V2 without prediction](https://drive.google.com/file/d/1_-C4MfUeC-6MXPDZlx6LTM48Tl8gdZpR/view?usp=sharing)
)

Please refer [OpenDAIRV2X](https://github.com/AIR-THU/DAIR-V2X/tree/main/configs/vic3d/middle-fusion-pointcloud/ffnet/README.md) for evaluating FFNet with OpenDAIRV2X. 

Example: evaluate ```FFNET``` on ```DAIR-V2X-C-Example``` with 100ms latency:

```
# modify the DATA to point to DAIR-V2X-C-Example in script ${OpenDAIRV2X_root}/v2x/scripts/lidar_feature_flow.sh
# bash scripts/lidar_feature_flow.sh [YOUR_CUDA_DEVICE] [YOUR_FFNET_WORKDIR] [DELAY_K] 
cd ${OpenDAIRV2X_root}/v2x
bash scripts/lidar_feature_flow.sh 0 /home/yuhaibao/FFNet-VIC3D 1
```

## Training

Firstly, train the basemodel on ```DAIR-V2X``` without latency
```
# Single-gpu training
cd ${FFNET-VIC_repo}
export PYTHONPATH=$PYTHONPATH:./
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/config_basemodel.py
```

Secondly, put the trained basemodel in a folder ```ffnet_work_dir/pretrained-checkpoints```.

Thirdly, train ```FFNET``` on ```DAIR-V2X``` with latency

```
# Single-gpu training
cd ${FFNET-VIC_repo}
export PYTHONPATH=$PYTHONPATH:./
CUDA_VISIBLE_DEVICES=$1 python tools/train.py configs/config_ffnet.py
```

## Citation
```latex
@inproceedings{yu2023ffnet,
  title={Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection},
  author={Yu, Haibao and Tang, Yingjuan and Xie, Enze and Mao, Jilei and Luo, Ping and Nie, Zaiqing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

```latex
@inproceedings{yu2023ffnet,
  title={Vehicle-Infrastructure Cooperative 3D Object Detection via Feature Flow Prediction},
  author={Yu, Haibao and Tang, Yingjuan and Xie, Enze and Mao, Jilei and Yuan, Jirui and Luo, Ping and Nie, Zaiqing},
  booktitle={https://arxiv.org/abs/2303.10552},
  year={2023}
}
```
