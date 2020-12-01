# Learning Modulated Loss for Rotated Object Detection

## Abstract
This repo is based on [Learning Modulated Loss for Rotated Object Detection](https://arxiv.org/pdf/1911.08299.pdf), 
and it is completed by [Qianwen](https://github.com/Mrqianduoduo/).

Thanks for yangxue(https://github.com/yangxue0827/) who helps me a lot.

## Performance
### DOTA1.0
mAP: 0.6687058601615324
ap of each class: plane:0.8878331545311091, baseball-diamond:0.6962231464499975, bridge:0.44458338981056794, ground-track-field:0.6432950394052023, small-vehicle:0.6795123578210454, large-vehicle:0.6020451193878097, ship:0.7612468381585155, tennis-court:0.90845010252905, basketball-court:0.7784977406333061, storage-tank:0.7638777969030915, soccer-ball-field:0.5521381565847773, roundabout:0.6047706037636372, harbor:0.5988219351889964, swimming-pool:0.6377906405858327, helicopter:0.47150188067004956

### model zoo
we propose our baseline model with out any data augmentation and refinement.


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
(2) Add category information in $PATH_ROOT/libs/label_name_dict/lable_dict.py     
(3) Add data_name to $PATH_ROOT/data/io/read_tfrecord.py 
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

3、multi-gpu train
```  
cd $PATH_ROOT/tools
python multi_gpu_train.py
```

## Eval
```  
cd $PATH_ROOT/tools
python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                    --gpus=0,1,2,3,4,5,6,7          
```  
