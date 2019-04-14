
# RoadNet example

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