# 1. Introduction
## Overview
![faster rcnn framework](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/01a86510-75c0-4247-aa76-755167272978)
\
\
Faster RCNN은 image feature extractor인 backbone CNN, target object의 클래스 분류, bounding regression을 위한 RPN, ROI Pooling의 3가지로 구성되어 있습니다.
전체 프로세스를 살펴보면 입력 이미지를 backbone CNN에 입력하여 feature map을 얻고 해당 feature map을 RPN의 입력으로 사용하여 candidate bounding box를 만들고 object의 존재 여부에 대한 binary classification을 수행합니다.
candidate bounding box을 feature map에서 추출하여 해당 영역에 대하여 ROI pooling을 수행하고, FC Layer로 전달하여 object가 있다고 판단한 box에 대한 multi-class classification 및 bounding box regression을 수행합니다.
## Backbone CNN
Faster RCNN 논문에서는 VGG와 ZFNet을 image feature extractor로 사용합니다. 여기서는 VGG와 ResNet50FPN backbone을 이용하여 모델을 학습하고 qualitative, quantitative evaluation을 수행합니다.
## Region Proposal Network (RPN)
![rpn](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/4f76f137-e629-4bab-b623-d1a2f7554302)
\
\
RPN은 backbone CNN의 출력인 feature map을 입력으로 받습니다. 위 그림처럼 7x7 size의 feature map이 있다고 가정하면, 3x3의 kernel size로 sliding window을 수행하여 모든 grid cell마다 서로 다른 크기의 $k$개의 Anchor Box를 정의합니다.
논문에서는 Anchor Box를 9개로 정의하지만 Anchor Box와 GT Box의 차이를 Regression으로 prediction하므로 다양한 크기의 predicted Box가 산출됩니다.
구현에서 (C, 7, 7) shape의 feature map을 3x3 convolution에 padding을 1로 설정하여 (256, 7, 7) size로 만듭니다. 
각 sliding window별로 object가 존재하는지, 하지 않는지에 대한 binary classification을 수행하기 위해 1x1 convolution을 이용하여 2(foreground/background) X $k$ = $2k$ channel로 만들고,
Box의 좌표에 대한 regression을 수행하기 위하여 1x1 convolution을 이용하여 4(x, y, w, h) X $k$ = $4k$의 channel로 만들고 각 channel axis에 대하여 binary classification, bounding box regression을 수행합니다.
## ROI Pooling
![rooi](https://github.com/cjw94103/CycleGAN_reimpl/assets/45551860/b364280f-bf45-4c83-8224-95591602749c)
\
\
ROI Pooling의 목적은 size가 다른 region proposal을 FC layer의 입력으로 사용하기 위해 fixed size의 feature로 만들기 위하여 사용됩니다.
Faster RCNN에서 사용한 ROI Pooling은 Fast RCNN에서 사용한 방법과 동일하며 사전 정의된 size의 grid를 이용하여 grid의 bin안에 들어가는 값 사이에 max pooling을 수행하여 fixed size의 feature vector를 만듭니다.
# 2. Dataset Preparation
데이터셋은 coco2017을 사용합니다. 아래의 명령어를 이용하여 데이터셋을 다운로드 해주세요.
```python
$ wget http://images.cocodataset.org/zips/train2017.zip   tree -L 2
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip

$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```
학습을 위한 데이터셋의 구조는 아래와 같습니다.
```python
└── coco2017
    ├── annotations
    ├── train
    └── val
```
annotations 폴더에는 train_annotations.json, val_annotations.json 파일을 위치시켜주세요. train, val 폴더에는 학습에 사용 할 이미지 파일이 있습니다.
