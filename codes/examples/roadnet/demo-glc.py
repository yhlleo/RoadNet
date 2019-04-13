import cv2
import sys
import tensorflow as tf
import argparse
import numpy as np
import os, glob
import time

sys.path.insert(0, '../../tensorpack')

from tensorpack import *
from roadnet_glc import Model
#from raw.segnet import Model
from imcrop import imageCrop
from imstitch import stitch

def im_eval(im_file, model_path, save_path):
  assert os.path.isfile(im_file), im_file
  if not os.path.exists(save_path):
    os.mkdir(save_path)

  dirname = os.path.dirname(im_file)
  basename = os.path.basename(im_file).split('.')[0]
  
  # final prediction map
  save_file = os.path.join(save_path, basename+'.png')
  im_raw = cv2.imread(im_file, cv2.IMREAD_UNCHANGED)
  assert im_raw is not None
  print "Loading image successful ..."
  
  # crop
  crop_save_path = os.path.join(dirname, basename)
  if not os.path.isdir(crop_save_path):
    os.mkdir(crop_save_path)
    imageCrop(im_file, crop_save_path)
  
  # model config
  pred_config = PredictConfig(
    model=Model(),
    session_init=get_model_loader(model_path),
    input_names=['image'],
    output_names=['segment-output5', 'skeleton-output4'])
  predictor = OfflinePredictor(pred_config)
  
  imgs = glob.glob(os.path.join(crop_save_path, '*.png'))
  time_consum = []
  
  output_path = os.path.join(save_path, 'output')
  if not os.path.isdir(output_path):
    os.mkdir(output_path)
  
  print 'Beginning predict image patches ...'
  print 'Total image patches: {}'.format(len(imgs))
  
  for ls in imgs:
    cur_im = cv2.imread(ls, cv2.IMREAD_UNCHANGED)
    assert cur_im is not None
    cur_im = cv2.resize(cur_im, (cur_im.shape[1] // 16 * 16, cur_im.shape[0] // 16 * 16))
	
    start = time.clock()
    outputs = predictor([[cur_im.astype('float32')]])
    end = time.clock()
    time_consum.append(end-start)
	
    fname = ls.split('/')[-1]
    fname = fname.split('.')[0]
    mask = cv2.merge([outputs[0][0], outputs[1][0], outputs[1][0]])
    cv2.imwrite(os.path.join(output_path,fname+'.png'), mask*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

  print 'Averaging infer time (s): %.3f'%(sum(time_consum)/len(time_consum)) 
  print 'Starting stitching image patches ...'
  panorama = stitch(crop_save_path, output_path)
  cv2.imwrite(save_file, panorama, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	
	
if __name__ == '__main__':
  #idxs = ['1', '16', '17', '18', '19', '20']
  for i in xrange(195, 225):
    print i
    im_file = "/home/tensorflow/yhl/tensorpack_data/Guangliang/test/image/image{}.jpg".format(str(i))
    model_path = "/home/tensorflow/yhl/tensorpack/examples/train_log/roadnet_glc/model-67600.data-00000-of-00001"
    save_path = '/home/tensorflow/yhl/tensorpack_data/Guangliang/test/pred/{}'.format(str(i))
    im_eval(im_file, model_path, save_path)
	
