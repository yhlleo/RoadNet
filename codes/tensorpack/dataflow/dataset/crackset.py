#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bsds500.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import glob
import cv2
import numpy as np

from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ['CrackImage']
IMG_W, IMG_H = 544, 384

class CrackImage(RNGDataFlow):
    """
    `Berkeley Segmentation Data Set and Benchmarks 500 dataset
    <http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>`_.

    Produce ``(image, label)`` pair, where ``image`` has shape (321, 481, 3(BGR)) and
    ranges in [0,255].
    ``Label`` is a floating point image of shape (321, 481) in range [0, 1].
    The value of each pixel is ``number of times it is annotated as edge / total number of annotators for this image``.
    """

    def __init__(self, name, shuffle=True):
        """
        Args:
            name (str): 'train', 'test', 'val'
            data_dir (str): a directory containing the original 'BSR' directory.
        """
        # check and download data
        name = 'train'
        self.data_root = '/home/swoda/tensorpack_data/crack_data'
        self.shuffle = shuffle
        self._load(name)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, 'images', '*.jpg')
        image_files = glob.glob(image_glob)
        gt_dir = os.path.join(self.data_root, 'labels2' )
        self.data = np.zeros((len(image_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.label = np.zeros((len(image_files), IMG_H, IMG_W), dtype='uint8')

        for idx, f in enumerate(image_files):
            #print f
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            assert im is not None
            if im.shape[0] > im.shape[1]:
                im = np.transpose(im, (1, 0, 2))
            assert im.shape[:2] == (IMG_H, IMG_W), "{} != {}".format(im.shape[:2], (IMG_H, IMG_W))

            imgid = os.path.basename(f).split('.')[0]
            gt_file = os.path.join(gt_dir, imgid+'.png')
            gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            gt = gt.astype('float32') / 255.0
            if gt.shape[0] > gt.shape[1]:
                gt = gt.transpose()
            assert gt.shape == (IMG_H, IMG_W)

            self.data[idx]  = im
            self.label[idx] = gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.label[k]]


try:
    from scipy.io import loadmat
except ImportError:
    from ...utils.develop import create_dummy_class
    CrackImage = create_dummy_class('CrackImage', 'scipy.io')  # noqa

if __name__ == '__main__':
    a = CrackImage('val')
    for k in a.get_data():
        cv2.imshow("haha", k[1].astype('uint8') * 255)
        cv2.waitKey(1000)
