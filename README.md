## Dataset



Benchmakr Dataset for the paper: [RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes from High-Resolution Remotely Sensed Images](https://ieeexplore.ieee.org/document/8506600), IEEE Transactions on Geoscience and Remote Sensing, 2018.

![dataset](./roadnet-dataset.jpg)

---------

We collected several typical urban areas of Ottawa, Canada from [Google Earth](http://earth.google.com). The images are with 0.21m spatial resolution per pixel.

### 1.Download

Download link: 

 - [BaiduYun](https://pan.baidu.com/s/1l9RZvyYfLgTOx_k4LQRyhQ)（Password: h2zt）
 - [GoogleDrive](https://drive.google.com/open?id=1GDHy7uwgOswuCDC49OamlNkAxjaITPBI)

### 2.Training and Testing

Training files:

 - 2,3,4,5,6,7,8,9,10,11,12,13,14,15

Testing files:

 - 1,16,17,18,19,20

### 3.Annotations

We take an example with the folder "1": 

|Filename|Explaination|
|:----:|:----|
|`Ottawa-1.tif`|original image|
|`segmentation.png`|manual annotaion of road surface|
|`edge.png`|manual annotation of road edge|
|`centerline.png`|manual annotation of road centerline|
|`extra.png`|roughly mark the heterogeneous regions with a single pixel width brush (red)|
|`extra-Ottawa-1.tif`| the `Ottawa-1.tif` is overlaid with the `extra.png`|

### 4.Citation

Please cite this paper if you use this dataset:

```
@article{liu2018roadnet,
title={RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes from High-Resolution Remotely Sensed Images},
author={Yahui, Liu and Jian, Yao and Xiaohu, Lu and Menghan, Xia and Xingbo, Wang and Yuan, Liu},
journal={IEEE Transactions on Geoscience and Remote Sensing},
year={2018},
}
```

You are also welcomed to contact me: yahui.cvrs@gmail.com
