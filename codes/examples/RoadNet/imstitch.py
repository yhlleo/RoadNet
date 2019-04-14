import cv2, os
import numpy as np

W, H = 512, 512
STEP_H, STEP_W = 256, 256

def imoverlap(im1, im2, overlap=(STEP_H, STEP_W)):
	assert im1.shape == im2.shape
	sz = im1.shape
	im1=im1.astype('float32')
	im2=im2.astype('float32')
	newh, neww = sz[0]*2 - STEP_H, sz[1]*2 - STEP_W
	merge = np.zeros((sz[0], neww, 3), dtype=np.uint8)
	merge[:, :sz[1]-STEP_W, :] = im1[:, :sz[1]-STEP_W, :]
	merge[:, sz[1]:, :] = im2[:, STEP_W:, :]
	merge[:,sz[1]-STEP_W:sz[1],:] = ((im1[:,sz[1]-STEP_W:sz[1],:]
		+im2[:,:STEP_W,:])/2).astype('uint8')
	return merge

def imload(im_folder, prefix, r, c):
	fname = os.path.join(im_folder, prefix+'{}-{}.png'.format(r,c))
	return cv2.imread(fname, cv2.IMREAD_UNCHANGED)

def warpper(im_patches, mode='bilinear', orit='H'):
	if mode == 'bilinear':
		return bilinear_warp(im_patches, orit=orit)
	else:
		return average_warp(im_patches)

def average_warp(im_patches):
	num = len(im_patches)
	assert num == 2 or num == 4, num
	mask = np.zeros(im_patches[0].shape)
	for i in range(num):
		mask += im_patches[i]
	return np.clip(mask/num, 0, 255).astype('uint8')

def bilinear_warp(im_patches, orit='H'):
	num = len(im_patches)
	assert num == 2 or  num == 4, num
	win = im_patches[0].shape
	mask = np.zeros(win[:2])
	if num == 2:
		if orit == 'H':
			for i in range(win[1]):
				mask[:,i] = float(i)/win[1]
		else:
			for i in range(win[0]):
				mask[i, :] = float(i)/win[0]
		if im_patches[0].shape > 2:
			mask = cv2.merge([mask, mask, mask])
		warped = (im_patches[0]*(1.0-mask) + im_patches[1]*mask)
		return np.clip(warped, 0, 255).astype('uint8')
	else:
		for i in range(win[0]):
			for j in range(win[1]):
				mask[i,j] = float((win[0]-i)*(win[1]-j))/(win[0]*win[1])
		if im_patches[0].shape > 2:
			mask = cv2.merge([mask, mask, mask])
		warped = (im_patches[0]*mask + im_patches[1]*np.fliplr(mask) +
			im_patches[2]*np.flipud(np.fliplr(mask)) + 
			im_patches[3]*np.flipud(mask))
		return np.clip(warped, 0, 255).astype('uint8')


def load_stitch_info(im_folder):
	fp = open(os.path.join(im_folder, 'info')).read().splitlines()
	return [int(i) for i in fp[0].split(' ')]

def center_stitch(im_folder, pred_folder, prefix=''):
	r, c, off_h, off_w = load_stitch_info(im_folder)#info
	newh = r*H - STEP_H*(r-1) - off_h
	neww = c*W - STEP_W*(c-1) - off_w

	finalmap = np.zeros((newh, neww, 3), dtype=np.uint8)

	half_step_h, half_step_w = STEP_H/2, STEP_W/2

	for i in range(r):
		for j in range(c):
			im_ij = imload(pred_folder, prefix, i, j)
			
			if (i==0 and j==0):
				finalmap[:half_step_h+STEP_H, :half_step_w+STEP_W] = im_ij[:half_step_h+STEP_H, :half_step_w+STEP_W]
			elif (i==0 and j==c-1):
				finalmap[:half_step_h+STEP_H, neww-(W-off_w):] = im_ij[:half_step_h+STEP_H, off_w:]
			elif (i==r-1 and j==0):
				finalmap[newh-(H-off_h):, :half_step_w+STEP_W] = im_ij[off_h:, :half_step_w+STEP_W]
			elif (i==r-1 and j==c-1):
				finalmap[newh-(H-off_h):, neww-(W-off_w):] = im_ij[off_h:, off_w:]
			else:
				if i==0:
					finalmap[:half_step_h, j*STEP_W+half_step_w:(j+1)*STEP_W+half_step_w] = im_ij[:half_step_h, half_step_w:STEP_W+half_step_w]
					finalmap[i*STEP_H+half_step_h:(i+1)*STEP_H+half_step_h, j*STEP_W+half_step_w:(j+1)*STEP_W+half_step_w] = im_ij[half_step_h:STEP_H+half_step_h, half_step_w:STEP_W+half_step_w]
				elif i==r-1:
					finalmap[newh-H:, j*STEP_W+half_step_w:(j+1)*STEP_W+half_step_w] = im_ij[:, half_step_w:STEP_W+half_step_w]
				elif j==0:
					finalmap[i*STEP_H+half_step_h:(i+1)*STEP_H+half_step_h, :half_step_w] = im_ij[half_step_h:STEP_H+half_step_h, :half_step_w]
					finalmap[i*STEP_H+half_step_h:(i+1)*STEP_H+half_step_h, j*STEP_W+half_step_w:(j+1)*STEP_W+half_step_w] = im_ij[half_step_h:STEP_H+half_step_h, half_step_w:STEP_W+half_step_w]
				elif j==c-1:
					finalmap[i*STEP_H+half_step_h:(i+1)*STEP_H+half_step_h, neww-W:] = im_ij[half_step_h:STEP_H+half_step_h, :]
				else:
					finalmap[i*STEP_H+half_step_h:(i+1)*STEP_H+half_step_h, j*STEP_W+half_step_w:(j+1)*STEP_W+half_step_w] = im_ij[half_step_h:STEP_H+half_step_h, half_step_w:STEP_W+half_step_w]
	return finalmap


def stitch(im_folder, pred_folder, mode='bilinear', prefix=''):
	r, c, off_h, off_w = load_stitch_info(im_folder)#info
	newh = r*H - STEP_H*(r-1) - off_h
	neww = c*W - STEP_W*(c-1) - off_w

	finalmap = np.zeros((newh, neww, 3), dtype=np.uint8)
	for i in range(r):
		for j in range(c):
			im_ij = imload(pred_folder, prefix, i, j)

			if (i==0 and j==0):
				finalmap[:H - STEP_H,:W - STEP_W,:] = im_ij[:H - STEP_H,:W - STEP_W,:]
			elif (i==0 and j==c-1):
				finalmap[:H - STEP_H, neww-(W-off_w):,:] = im_ij[:H-STEP_H, off_w:,:]
			elif (i==r-1 and j==0):
				finalmap[newh-(H-off_h):, :W - STEP_W,:] = im_ij[off_h:, :W-STEP_W,:]
			elif (i==r-1 and j==c-1):
				finalmap[newh-(H-off_h):, neww-(W-off_w):,:] = im_ij[off_h:, off_w:,:]
			else:
				if i==0:
					im_ij_l = imload(pred_folder, prefix, i, j-1)

					finalmap[:H - STEP_H, j*STEP_W:(j+1)*STEP_W,:] = warpper(
						[im_ij_l[:H - STEP_H,W - STEP_W:], im_ij[:H - STEP_H,:W - STEP_W]], 
						mode=mode,
						orit='H')
				elif i==r-1:
					im_ij_l = imload(pred_folder, prefix, i, j-1)

					finalmap[newh-H:, j*STEP_W:(j+1)*STEP_W,:] = warpper(
						[im_ij_l[:,W - STEP_W:], im_ij[:,:W - STEP_W]], 
						mode=mode,
						orit='H')					
				elif j==0:
					im_ij_u = imload(pred_folder, prefix, i-1, j)

					finalmap[i*STEP_H:(i+1)*STEP_H, :W-STEP_W,:] = warpper(
						[im_ij_u[H - STEP_H:,:W - STEP_W], im_ij[:H - STEP_H,:W - STEP_W]], 
						mode=mode,
						orit='V')
				elif j==c-1:
					im_ij_u = imload(pred_folder, prefix, i-1, j)

					finalmap[i*STEP_H:(i+1)*STEP_H, neww-W:,:] = warpper(
						[im_ij_u[H - STEP_H:,:], im_ij[:H - STEP_H,:]], 
						mode=mode,
						orit='V')
				else:
					im_ij_0 = imload(pred_folder, prefix, i-1, j-1)
					im_ij_1 = imload(pred_folder, prefix, i-1, j)
					im_ij_3 = imload(pred_folder, prefix, i, j-1)
				
					finalmap[i*STEP_H:(i+1)*STEP_H, j*STEP_W:(j+1)*STEP_W,:] = warpper(
						[im_ij_0[H - STEP_H:,W - STEP_W:], 
						im_ij_1[H - STEP_H:,:W - STEP_W],
						im_ij[:H-STEP_H, :W-STEP_W],
						im_ij_3[:H-STEP_H, W-STEP_W:]],
						mode=mode)				
	return finalmap


if __name__ == '__main__':
	'''
	root_dir = '/home/guangxi/zly/tensorflow/yhl/tensorpack_data/RoadNet/Ottawa/19'
	im = stitch(
		os.path.join(root_dir, 'extra-Ottawa-19'), 
		os.path.join(root_dir, 'plusplus'),
    mode='average')

	print im.shape
	cv2.imwrite(os.path.join(root_dir, 'Ottawa-19-avg.png'), 
		im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	print "OK"
	'''
	'''
	root_dir = '/home/guangxi/zly/tensorflow/yhl/tensorpack_data/RoadNet/Ottawa/1'
	im = center_stitch(
		os.path.join(root_dir, 'image'), 
		os.path.join(root_dir, 'plusplus'))
	'''

	root_dir = '/home/guangxi/zly/tensorflow/yhl/tensorpack_data/RoadNet/interner'
	import time
	tbegin = time.clock()
	im = stitch(
		os.path.join(root_dir, 'Guanggu'),
		os.path.join(root_dir, 'G-output'),
	)
	tend = time.clock()
	print im.shape, tbegin-tend
	cv2.imwrite(os.path.join(root_dir, 'Guanggu-roadnet.png'), 
		im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	print "OK"