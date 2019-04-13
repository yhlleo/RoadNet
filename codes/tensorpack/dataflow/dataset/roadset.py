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

__all__ = ['RoadNetImage']
IMG_W, IMG_H = 512, 512

class RoadNetImage(RNGDataFlow):
    """

    Produce ``(image, segment, boundary, skeleton)`` pair, where ``image`` has shape (512, 512, 3(BGR)) and
    ranges in [0,255].
    ``segment, boundary and skeleton`` are floating point images of shape (512, 512) in range [0, 1].
    The value of each pixel is ``number of times it is annotated as edge / total number of annotators for this image``.
    """

    def __init__(self, name, im_dir, shuffle=True):
        """
        Args:
            name (str): 'train', 'test', 'val'
            data_dir (str): a directory containing the original 'BSR' directory.
        """
        # check and download data
        name = 'train'
        self.data_root = im_dir
        self.shuffle = shuffle
        self._load(name)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, 'image', '*.png')
        image_files = glob.glob(image_glob)
        segment_dir = os.path.join(self.data_root, 'segment' )
        boundary_dir = os.path.join(self.data_root, 'edge' )
        skeleton_dir = os.path.join(self.data_root, 'skeleton' )

        self.data = np.zeros((len(image_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.segment = np.zeros((len(image_files), IMG_H, IMG_W), dtype='uint8')
        self.boundary = np.zeros((len(image_files), IMG_H, IMG_W), dtype='uint8')
        self.skeleton = np.zeros((len(image_files), IMG_H, IMG_W), dtype='uint8')

        for idx, f in enumerate(image_files):
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            assert im is not None
            if im.shape[0] > im.shape[1]:
                im = np.transpose(im, (1, 0, 2))
            assert im.shape[:2] == (IMG_H, IMG_W), "{} != {}".format(im.shape[:2], (IMG_H, IMG_W))

            imgid = os.path.basename(f).split('.')[0]

            segment_file = os.path.join(segment_dir, imgid+'.png')
            segment = cv2.imread(segment_file, cv2.IMREAD_UNCHANGED)

            boundary_file = os.path.join(boundary_dir, imgid+'.png')
            boundary = cv2.imread(boundary_file, cv2.IMREAD_UNCHANGED)

            skeleton_file = os.path.join(skeleton_dir, imgid+'.png')
            skeleton = cv2.imread(skeleton_file, cv2.IMREAD_UNCHANGED)

            segment = segment.astype('float32') / 255.0
            boundary = boundary.astype('float32') / 255.0
            skeleton = skeleton.astype('float32') / 255.0

            if segment.shape[0] > segment.shape[1]:
                segment = segment.transpose()
            if boundary.shape[0] > boundary.shape[1]:
                boundary = boundary.transpose()
            if skeleton.shape[0] > skeleton.shape[1]:
                skeleton = skeleton.transpose()

            assert segment.shape == (IMG_H, IMG_W)
            assert boundary.shape == (IMG_H, IMG_W)
            assert skeleton.shape == (IMG_H, IMG_W)

            self.data[idx]  = im
            self.segment[idx] = segment
            self.boundary[idx] = boundary
            self.skeleton[idx] = skeleton

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.segment[k], self.boundary[k], self.skeleton[k]]


try:
    from scipy.io import loadmat
except ImportError:
    from ...utils.develop import create_dummy_class
    RoadNetImage = create_dummy_class('RoadNetImage', 'scipy.io')  # noqa

if __name__ == '__main__':
    a = RoadNetImage('val')
    for k in a.get_data():
        cv2.imshow("haha", k[1].astype('uint8') * 255)
        cv2.waitKey(1000)
