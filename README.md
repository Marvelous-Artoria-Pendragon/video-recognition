# pytorch-video-recognition

<table style="border:0px">
   <tr>
       <td><img src="assets/demo1.gif" frame=void rules=none></td>
       <td><img src="assets/demo2.gif" frame=void rules=none></td>
   </tr>
</table>

## Introduction
该项目在项目https://github.com/jfzhang95/pytorch-video-recognition.git上做了一定修改，集合了视频行为识别的几种模型，包括C3D, R(2+1)D, R3D(效果不太好), Resnet, I3D, GBDT和XGBoost(分类器)，并基于PyTorch 1.10.0及以上版本运行

## Installation
代码在Anaconda和Python 3.6 及以上版本运行。在配置完 Anaconda 环境后：


1. 克隆项目:
    ```Shell
    git clone https://github.com/Marvelous-Artoria-Pendragon/video-recognition.git
    cd pytorch-video-recognition
    ```

2. 安装依赖库:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install opencv-python==3.4.2.17
    pip install tqdm scikit-learn tensorboardX
    pip install argparse
    pip install joblib
    ```

3. 如有需求，自行下载C3D预训练模型(c3d-pretrained.pth)、I3D预训练模型(flow_charades.pt, flow_imagenet.pt, rgb_charades.pt, rgb_imagenet.pt)

4. 到这里，原项目需要配置mypath指定文件路径，本项目设定命令行传参运行，免去每次修改参数的麻烦。这里提供了CIBR-14的部分数据集 CIBR-2 供测试训练：

    训练一个I3D模型，在命令行输入:
    ```Shell
    python train.py --model I3D --epoch 5 --useTest --nTestInterval 5 --snapshot 5 --lr 1e-3 --n_frame 16 --dataset tb --data_dir ./CIBR-2 --save_dir ./out --label_dir ./dataloaders --n_class 2 --height 226 --width 226 --crop_size 224 --batch_size 1 --n_worker 0
    ```

    使用I3D提取特征(目前只支持)：
    ```Shell
    python extract_feature.py --dataset tb --n_class 2 --model_path ./out/models/I3D-tb_epoch-4.pth.tar --data_path ./CIBR-2 --num_frame 16 --save_dir ./out --batch_size 1 --height 226 --width 226 --crop_size 224
    ```
    
    对视频进行分类预测：
    ```Shell
    python inference.py --model I3D --n_class 2 --check_point ./out/models/I3D-tb_epoch-4.pth.tar --label_path ./dataloaders/tb_labels.txt --video_dir ./CIBR-2/opbb --output_dir ./out  --n_frame 16
    ```

5. 其它说明：
    获取参数说明, 如：
    ```Shell
    python train.py -h
    python extract_feature.py -h
    python inference.py -h
    ```
  

## Datasets:

项目中我使用了自己的数据集CIBR-14训练，当然也可以使用UCF101和HMD51

数据集文件结构样例如下：

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

Note: HMDB dataset's directory tree is similar to UCF101 dataset's.

## Experiments
CIBR-14数据集是在Kaggle上训练的，配置为：
    CPU：双核的Intel(R) Xeon(R) CPU @ 2.00GHz 13GB
    GPU: 两块Tesla T4 14.8GB

若机器硬件不行，请适当降低参数batch_size大小

