# Learning Modulated Loss for Rotated Object Detection

[![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A1911.08299-B31B1B.svg)](https://arxiv.org/abs/1911.08299)

## Abstract
This is a tensorflow-based rotation detection method, called RSDet. 
RSDet is completed by [Qianwen](https://github.com/Mrqianduoduo/).

Thanks for [yangxue](https://github.com/yangxue0827/) who helps me a lot.
And we also provide a modified version in rotation detection [benchmark](https://github.com/yangxue0827/RotationDetection), which achieve better perfirmance.

## Performance
### DOTA1.0
| Model |  Neck  |    Backbone    |    Training/test dataset    |    mAP   | Model Link | Anchor | Angle Pred. | Reg. Loss| Angle Range | Data Augmentation | Configs |      
|:------------:|:------------:|:------------:|:------------:|:-----------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------:|:---------:|    
| [RetinaNet-H](https://arxiv.org/abs/1908.05612) | FPN | ResNet50_v1d 600->800 | DOTA1.0 trainval/test | 64.17 | [Baidu Drive (j5l0)](https://pan.baidu.com/s/1Qh_LE6QeGsOBYqMzjAESsA) | H | Reg. | smooth L1 | **180** | × | - |
| [RetinaNet-H](https://arxiv.org/abs/1908.05612) | FPN | ResNet50_v1d 600->800 | DOTA1.0 trainval/test | 65.73 | [Baidu Drive (jum2)](https://pan.baidu.com/s/19-hEtCGxLfYuluTATQJpdg) | H | Reg. | smooth L1 | **90** | × | - |
| [RSDet](https://arxiv.org/pdf/1911.08299) | FPN | ResNet50_v1d 600->800 | DOTA1.0 trainval/test | 66.87 | [Baidu Drive (6nt5)](https://pan.baidu.com/s/1-4iXqRMvCOIEtrMFwtXyew) | H | Reg. | modulated loss | - | × | [cfgs_res50_dota_rsdet_v2.py](./libs/configs/DOTA/rsdet/cfgs_res50_dota_rsdet_v2.py) |

## My Development Environment
**docker images: docker pull yangxue2docker/yx-tf-det:tensorflow1.13.1-cuda10-gpu-py3**      
1. python3.5 (anaconda recommend)               
2. cuda 10.0                     
3. [opencv(cv2)](https://pypi.org/project/opencv-python/)       
4. [tfplot 0.2.0](https://github.com/wookayin/tensorflow-plot) (optional)            
5. tensorflow-gpu 1.13
6. tqdm 4.54.0

## Download Model
### Pretrain weights
1. Please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)) pre-trained models on Imagenet, put them to $PATH_ROOT/dataloader/pretrained_weights.       
2. **(Recommend in this repo)** Or you can choose to use better backbones (resnet_v1d), refer to [gluon2TF](https://github.com/yangJirui/gluon2TF).    
* [Baidu Drive (b640)](https://pan.baidu.com/s/1lT9t6Lr7-xrmOijSB9QK-Q)              

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace (or make)

cd $PATH_ROOT/libs/box_utils/
python setup.py build_ext --inplace
```

## Train

1、If you want to train your own data, please note:  
```     
(1) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/cfgs.py
(2) Add category information in $PATH_ROOT/libs/label_name_dict/label_dict.py     
(3) Add data_name to $PATH_ROOT/data/io/read_tfrecord_multi_gpu.py
```     

2、make tfrecord     
For DOTA dataset:      
```  
cd $PATH_ROOT\data\io\DOTA
python data_crop.py
```  

```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/DOTA/' 
                                   --xml_dir='labeltxt'
                                   --image_dir='images'
                                   --save_name='train' 
                                   --img_format='.png' 
                                   --dataset='DOTA'
```      

3、Multi-gpu train
```  
cd $PATH_ROOT/tools
python multi_gpu_train.py
```

## Test
```  
cd $PATH_ROOT/tools
python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                    --gpus=0,1,2,3,4,5,6,7          
```  


## Citation

If you find our code useful for your research, please consider cite.
```
@article{qian2019learning,
    title={Learning modulated loss for rotated object detection},
    author={Qian, Wen and Yang, Xue and Peng, Silong and Guo, Yue and Yan, Chijun},
    journal={arXiv preprint arXiv:1911.08299},
    year={2019}
}
```

