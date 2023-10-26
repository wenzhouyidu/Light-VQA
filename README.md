# Light-VQA
Light_VQA: A Multi-Dimensional Quality Assessment Model for Low Light Video Enhancement
## Description
This is a repository for the model proposed in the paper ["Light_VQA: A Multi-Dimensional Quality Assessment Model for Low Light Video Enhancement"](https://arxiv.org/abs/2305.09512)(accepted by ACM MM 2023).
![image](https://github.com/wenzhouyidu/Light-VQA/blob/master/framework.png)
The framework of Light-VQA is illustrated in here. Considering
that among low-level features, brightness and noise have the most
impact on low-light enhanced VQA [37], in addition to semantic
features and motion features extracted from deep neural network,
we specially handcraft the brightness, brightness consistency, and
noise features to improve the ability of the model to represent the
quality-aware features of low-light enhanced videos. Extensive
experiments validate the effectiveness of our network design.

## Usage

### Install Requirements
```
pytorch
opencv
scipy
pandas
torchvision
torchvideo
```

### Download databases
[LLVE-QA(1)](https://drive.google.com/file/d/1eHWxZ7za-GwwtS_JKQjHcLni4dUXv6HT/view?usp=drive_link)
[LLVE-QA(2)](https://drive.google.com/drive/folders/1cbl7ZCNsgfYlo_41ypfZELaXZlwOYdsT?usp=sharing)

### Train models
1. Extract key frames (Set the file path internally)
```shell
python extract_key_frames.py
```
2. Extract brightness consistency features
```shell
python brightness_consistency.py
```
3. Extract temporal features
```shell
python extract_temporal_features.py
```
4. Train the model
```shell
python train.py
```
4. Test the model
```shell
python test.py
```
Pretrained weights can be downloaded here: https://drive.google.com/file/d/1GEvjpbDwG7L3fekkLt2eQQ3ozzAz3qCx/view?usp=sharing.

## Citation

If you find this code is useful for your research, please cite:
```
@inproceedings{lightvqa2023,
title = {Light_VQA: A Multi-Dimensional Quality Assessment Model for Low Light Video Enhancement},
author = {Yunlong Dong, Xiaohong Liu, Yixuan Gao, Xunchu Zhou, Tao Tan and Guangtao, Zhai},
booktitle={Proceedings of the 31th ACM International Conference on Multimedia},
year = {2023},
}
```
