整理图像、语音类型的数据增强方法、工具、论文

# 图像分类数据增强
## 原始数据结构
```
data
|__train
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__val
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__test
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
```

### c* 表示类别文件夹，包含属于该类别的图像数据
## 增强后数据结构
```
data_aug
|__train
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__val
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
|__test
    |__c1
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c2
        |__*.png/*.jpg/*.bmp/*.tiff
    |__c3
        |__*.png/*.jpg/*.bmp/*.tiff
    ...
    |__cn
        |__*.png/*.jpg/*.bmp/*.tiff
```

# 目标检测数据增强 (图像xyz.png 与标注文件xyz.txt 的名字要一致)

```
data
|__train
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__labels
        |__*.txt
|__val
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__labels
        |__*.txt
|__test
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__labels
        |__*.txt
```

### 其中*.txt 满足归一化后yolo标注格式: ID cw ch w h
### label 格式详解见[[各种格式](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)]
### default (albumentations) 归一化 [ID xmin ymin xmax ymax] 如 [2 0.1531 0.7187 0.6562 0.9625]
### voc (pascal_voc) 未归一化 [ID xmin ymin xmax ymax] 如 [2 98 345 420 462]
### coco 未归一化 [ID center_x center_y width height] 如 [2 98 345 322 117]
### yolo 归一化 [ID center_x center_y width height] 如 [2 0.4046 0.8406 0.5031 0.2437]

# 语义分割数据增强 (图像xyz.png 与标注文件xyz.jpg 的名字要一致)
```
data
|__train
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__masks
        |__*.png/*.jpg/*.bmp/*.tiff
|__val
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__masks
        |__*.png/*.jpg/*.bmp/*.tiff
|__test
    |__images
        |__*.png/*.jpg/*.bmp/*.tiff
    |__masks
        |__*.png/*.jpg/*.bmp/*.tiff   
```