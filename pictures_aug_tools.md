# 图像增强工具 
[参考](https://github.com/AgaMiko/data-augmentation-review/blob/master/README.md#Computer-vision)

序号 |Tools\/library| 开发单位 | 时间  |  是否开源  | 支持框架 | 增强方法 | 数据类型 | 数据格式 | 优点 | 综合评分 |
-----|--------------| --------| ----- | -----------| --------| --------| --------|  ------  | ------| ------|
| 1 |[Augmentor](https://github.com/mdbloice/Augmentor)| Marcus D.Bloice | 2017|   是   | TF、Keras、PyTorch | 7种类型基础增强方法 | 图像 | image |简单、易上手|:+1:|
| 2 |[Albumentations](https://github.com/albumentations-team/albumentations)| Albumentations.AI | 2020 |   是   | TF、Keras、Pytorch | 52 | 图像 | image、mask、bbox（部分支持）、keypoint（部分支持） |快速、方法多|:+1::+1::+1:|
| 3 |[ImgAug](https://github.com/aleju/imgaug)| MIT | 2020 |   是   | Pytorch | 70+ | 图像 | image、mask、bbox、keypoint、polygon、heatmat、line strings |支持数据格式多、方法多 |:+1::+1::+1::+1::+1:|
| 4 |[Kornia](https://github.com/kornia/kornia)| Apache 2.0 | 2020 |   是   | Pytorch | 23 | 图像 | image、mask、bbox、keypoint |可微、支持GPU、速度快 |:+1::+1:|
| 5 |[AugLy](https://github.com/facebookresearch/AugLy)| Facebook | 2022 |   是   | TF、PyTorch | 100+ | 图像、文本、语音 | image、mask、bbox、keypoint | 支持多种适用方式，如class-based、functional-based、numpy warper，添加文本、插入表情符号和截图叠加|:+1::+1::+1::+1::+1:|
| 6 |[Augraphy](https://github.com/sparkfish/augraphy)| Sparkfish LLC | 2023|   是   | TF、PyTorch | 1 | 图像文字识别 | image、mask、bbox、keypoint（部分支持） | 图像文字增强 |:+1::+1::+1:|

# 支持的增强方法
| 序号 |Transform Methods | Augmentor | Albumentation | Augly | ImgAug |
|-----|------------| ----------| ------------- | ------| --------|
| 1 |geometric transformations | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | :white_check_mark: |
| 2 |color adjustments    | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | :white_check_mark: |
| 3 |blurring    | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | :white_check_mark: |
| 4 |noise additions    | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | :white_check_mark: |
| 5 |<i>MixUP</i>| :heavy_multiplication_x: | :white_check_mark:  | :heavy_multiplication_x:  | :heavy_multiplication_x: |
| 6 |<i>CutMix</i>| :heavy_multiplication_x:| :white_check_mark:  | :heavy_multiplication_x:  | :heavy_multiplication_x: |
| 7 |<i>OverlayElements</i>| :heavy_multiplication_x: | :white_check_mark:  | :heavy_multiplication_x:  | :heavy_multiplication_x: |
| 8 |<i> OverLay*​</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :white_check_mark:  | :heavy_multiplication_x: |
| 9 |<i>elastic transformations</i> | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 10 |<i> polar warping​</i> | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 12 |<i> arithmetic​</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 13 |<i> artistic(Cartoon)​</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 14 |<i> blend</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 15 |<i> convolutional</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 16 |<i> edges(Canny)</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 17 |<i> pooling</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 18 |<i> segmentation</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 19 |<i> size</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
| 20 |<i> weather</i>  | :heavy_multiplication_x: | :heavy_multiplication_x:  | :heavy_multiplication_x:  | :white_check_mark: |
  
