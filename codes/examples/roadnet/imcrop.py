import cv2, os
import glob
import numpy as np

IMG_READ_MODE = cv2.IMREAD_UNCHANGED
PNG_SAVE_MODE = [cv2.IMWRITE_PNG_COMPRESSION, 0]
H, W = 512, 512
step = 256

def cropmode(im_shape, sz=(H,W), step=step):
  new_h = (im_shape[0] - step)/(sz[0] - step)
  offset_h = (im_shape[0] - step)%(sz[0] - step)
  if offset_h > 0:
     new_h += 1
     offset_h = step - offset_h
  new_w = (im_shape[1] - step)/(sz[1] - step)
  offset_w = (im_shape[1] -step)%(sz[1] - step)
  if offset_w > 0:
    new_w += 1
    offset_w = step - offset_w
  return new_h, new_w, offset_h, offset_w

def imageCrop(im_file, save_path):
  assert os.path.isdir(save_path)
  im = cv2.imread(im_file, IMG_READ_MODE)
  s = im.shape
  print(s)
  new_h, new_w, offset_h, offset_w = cropmode(s)
  fp = open(os.path.join(save_path, 'info'), 'w')
  fp.write(str(new_h)+' '+str(new_w)+' '+str(offset_h)+' '+str(offset_w))
  fp.close()
  print new_h, new_w, offset_h, offset_w
  h, w = 0, 0
  for i in range(new_h):
    h = i * (H - step)
    if i == new_h-1:
      h -= offset_h
    for j in range(new_w):
      w = j * (W - step)
      if j == new_w-1:
        w -= offset_w
      im_roi = im[h:h+H, w:w+W, :]
      cv2.imwrite(os.path.join(save_path, 
        "{}-{}.png".format(i,j)), 
        im_roi, PNG_SAVE_MODE)
      

if __name__ == '__main__':
  im_file = '/home/guangxi/zly/tensorflow/yhl/tensorpack_data/RoadNet/interner/Guanggu.png'
  save_path = '/home/guangxi/zly/tensorflow/yhl/tensorpack_data/RoadNet/interner/Guanggu'
  imageCrop(im_file, save_path)
