# RoadNet

---------

Code for [RoadNet: Learning to Comprehensively Analyze Road Networks in Complex Urban Scenes from High-Resolution Remotely Sensed Images](https://ieeexplore.ieee.org/document/8506600). Built based-on the open source project: [tensorpack: A Neural Net training Interface on TensorFlow](https://github.com/tensorpack/tensorpack). 

---------

## 1.Dataset

[More Details >>>](../README.md)

## 2.Training/Test

```
cd examples
```

 - Train

```
python ./RoadNet/roadnet.py \
  --data_dir ../datasets/Ottwa/train \
  --load_size 512 \
  --gpu 0
```

 - Test

```
python ./RoadNet/roadnet.py \
  --load /path/to/model_file \
  --run /path/to/images \
  --output /path/to/save_folder
```

 [More Details >>>](./examples/README.md)
