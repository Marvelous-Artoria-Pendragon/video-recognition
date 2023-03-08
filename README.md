# pytorch-video-recognition

<table style="border:0px">
   <tr>
       <td><img src="assets/demo1.gif" frame=void rules=none></td>
       <td><img src="assets/demo2.gif" frame=void rules=none></td>
   </tr>
</table>

## Introduction
This repo is an updated version from the previews, which contains several models for video action recognition,
including **C3D**, **R2Plus1D**, **R3D**, **I3D**, inplemented using PyTorch (0.4.0).
Currently, I train these models on UCF101 and HMDB51 datasets.
~~**More models and datasets will be available soon!**~~

**Note: An interesting online web game based on C3D model is in [here](https://github.com/jfzhang95/project-demo).**

## Installation
The code was tested with Anaconda and Python 3.7. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/Marvelous-Artoria-Pendragon/pytorch-video-recognition.git
    cd pytorch-video-recognition
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    conda install opencv
    pip install tqdm scikit-learn tensorboardX
    ```

2. Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw) or 
[GoogleDrive](https://drive.google.com/file/d/19NWziHWh1LgCcHU34geoKwYezAogv9fX/view?usp=sharing).
   Currently only support pretrained model for C3D.

3. Configure your dataset and pretrained model path in
[mypath.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/mypath.py).

4. You can choose different models and datasets in
[train.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/train.py).

    To train the model, please do:
    ```Shell
    python train.py
    ```

## Datasets:

There're two different datasets: UCF101 and HMDB.

Dataset directory tree is shown below

- **UCF101**
Make sure to put the files as the following structure:
  ```
  UCF-101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01.avi
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01.avi
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01.avi
  │   └── ...
  ```
After pre-processing, the output dir's structure is as follows:
  ```
  ucf101
  ├── ApplyEyeMakeup
  │   ├── v_ApplyEyeMakeup_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ├── ApplyLipstick
  │   ├── v_ApplyLipstick_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  └── Archery
  │   ├── v_Archery_g01_c01
  │   │   ├── 00001.jpg
  │   │   └── ...
  │   └── ...
  ```

Note: HMDB dataset's directory tree is similar to UCF101 dataset's. If you want to use your dataset, please modify the dateset's path on **mypthon.py**

## Experiments
(orginal words..) These models were trained in machine with NVIDIA TITAN X 12gb GPU. Note that I splited
train/val/test data for each dataset using sklearn. If you want to train models using
official train/val/test data, you can look in [dataset.py](https://github.com/Marvelous-Artoria-Pendragon/pytorch-video-recognition/blob/master/dataloaders/dataset.py), and modify it to your needs.

Currently, I only train C3D model in UCF and HMDB datasets. The train/val/test
accuracy and loss curves for each experiment are shown below:

- **UCF101**

<p align="center"><img src="assets/ucf101_results.png" align="center" width=900 height=auto/></p>

- **HMDB51**

<p align="center"><img src="assets/hmdb51_results.png" align="center" width=900 height=auto/></p>

~~Experiments for other models will be updated soon ...~~
