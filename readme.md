整理图像、语音类型的数据增强方法、工具、论文

## 图像分类数据增强
### 原始数据结构
```
dataset
|__train
    |__category01
        |__2011_003876.png/*.jpg/*.bmp/*.tiff
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
        |__2321_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__val
    |__category01
        |__2231_003876.png/*.jpg/*.bmp/*.tiff
        |__2232_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2234_003324.png/*.jpg/*.bmp/*.tiff
        |__2235_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__test
    |__category01
        |__3451_003876.png/*.jpg/*.bmp/*.tiff
        |__3461_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2543_003324.png/*.jpg/*.bmp/*.tiff
        |__2890_324089.png/*.jpg/*.bmp/*.tiff
    |__...
```

### category** 表示类别文件夹，包含属于该类别的图像数据

## 增强后数据结构
```
dataset_aug
|__train
    |__category01
        |__2011_003876.png/*.jpg/*.bmp/*.tiff
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2211_003324.png/*.jpg/*.bmp/*.tiff
        |__2321_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__val
    |__category01
        |__2231_003876.png/*.jpg/*.bmp/*.tiff
        |__2232_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2234_003324.png/*.jpg/*.bmp/*.tiff
        |__2235_324089.png/*.jpg/*.bmp/*.tiff
    |__...
|__test
    |__category01
        |__3451_003876.png/*.jpg/*.bmp/*.tiff
        |__3461_003324.png/*.jpg/*.bmp/*.tiff
    |__category02
        |__2543_003324.png/*.jpg/*.bmp/*.tiff
        |__2890_324089.png/*.jpg/*.bmp/*.tiff
    |__...
```

## 目标检测数据增强 (图像xyz.png 与标注文件xyz.txt 的名字要一致)

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

## 语义分割数据增强 (图像xyz.png 与标注文件xyz.jpg 的名字要一致)
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

#### RGB 图像：其mode属性值为'RGB'，表示红（Red）、绿（Green）、蓝（Blue）三个通道，每个通道代表一种基本颜色，通过不同比例混合这三种颜色来呈现丰富多彩的图像。

#### 灰度图像：mode属性值为'L'，它只有一个通道，像素值表示从黑（0）到白（255）的灰度级别。

#### 调色板模式（'P'）：这种模式下的图像通常是单通道，像素值是指向调色板的索引。调色板中存储了实际的 RGB 颜色值，通过索引从调色板获取颜色来显示图像，所以虽然从通道角度看类似单通道，但显示可能呈现彩色效果。