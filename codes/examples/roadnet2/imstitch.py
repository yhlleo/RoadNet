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

def imload(im_folder, r, c):
	fname = '{}-{}.png'.format(r,c)
	fname = os.path.join(im_folder, fname)
	return cv2.imread(fname, cv2.IMREAD_UNCHANGED)

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

def stitch(im_folder, pred_folder):
	r, c, off_h, off_w = load_stitch_info(im_folder)#info
	newh = r*H - STEP_H*(r-1) - off_h
	neww = c*W - STEP_W*(c-1) - off_w

	finalmap = np.zeros((newh, neww, 3), dtype=np.uint8)
	for i in range(r):
		for j in range(c):
			im_ij = cv2.imread(
				os.path.join(pred_folder, '{}-{}.png'.format(i,j)), 
				cv2.IMREAD_UNCHANGED)
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
					im_ij_l = cv2.imread(os.path.join(pred_folder, '{}-{}.png'.format(i,j-1)), 
						cv2.IMREAD_UNCHANGED)
					finalmap[:H - STEP_H, j*STEP_W:(j+1)*STEP_W,:] = bilinear_warp(
						[im_ij_l[:H - STEP_H,W - STEP_W:], im_ij[:H - STEP_H,:W - STEP_W]], orit='H')
				elif i==r-1:
					im_ij_l = cv2.imread(os.path.join(pred_folder, '{}-{}.png'.format(i,j-1)), 
						cv2.IMREAD_UNCHANGED)
					finalmap[newh-H:, j*STEP_W:(j+1)*STEP_W,:] = bilinear_warp(
						[im_ij_l[:,W - STEP_W:], im_ij[:,:W - STEP_W]], orit='H')					
				elif j==0:
					im_ij_u = cv2.imread(
						os.path.join(pred_folder, '{}-{}.png'.format(i-1,j)), 
						cv2.IMREAD_UNCHANGED)
					finalmap[i*STEP_H:(i+1)*STEP_H, :W-STEP_W,:] = bilinear_warp(
						[im_ij_u[H - STEP_H:,:W - STEP_W], im_ij[:H - STEP_H,:W - STEP_W]], orit='V')
				elif j==c-1:
					im_ij_u = cv2.imread(
						os.path.join(pred_folder, '{}-{}.png'.format(i-1,j)), 
						cv2.IMREAD_UNCHANGED)
					finalmap[i*STEP_H:(i+1)*STEP_H, neww-W:,:] = bilinear_warp(
						[im_ij_u[H - STEP_H:,:], im_ij[:H - STEP_H,:]], orit='V')
				else:
					im_ij_0 = cv2.imread(os.path.join(pred_folder, '{}-{}.png'.format(i-1,j-1)), 
						cv2.IMREAD_UNCHANGED)
					im_ij_1 = cv2.imread(os.path.join(pred_folder, '{}-{}.png'.format(i-1,j)), 
						cv2.IMREAD_UNCHANGED)
					im_ij_3 = cv2.imread(os.path.join(pred_folder, '{}-{}.png'.format(i,j-1)), 
						cv2.IMREAD_UNCHANGED)					
					finalmap[i*STEP_H:(i+1)*STEP_H, j*STEP_W:(j+1)*STEP_W,:] = bilinear_warp(
						[im_ij_0[H - STEP_H:,W - STEP_W:], 
						im_ij_1[H - STEP_H:,:W - STEP_W],
						im_ij[:H-STEP_H, :W-STEP_W],
						im_ij_3[:H-STEP_H, W-STEP_W:]])				
	return finalmap


if __name__ == '__main__':
	im = stitch('/home/tensorflow/yhl/tensorpack_data/RoadNet/Ottawa/1/casnet')
	print im.shape
	cv2.imwrite('/home/tensorflow/yhl/tensorpack_data/RoadNet/Ottawa/1/casnet.png', 
		im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	'''
	patches = []
	#im = np.ones((256, 256), dtype=np.float32)*255
	im1 = cv2.imread('./test-im/frrn/3-4.png', cv2.IMREAD_UNCHANGED)
	im2 = cv2.imread('./test-im/frrn/3-5.png', cv2.IMREAD_UNCHANGED)
	im3 = cv2.imread('./test-im/frrn/4-4.png', cv2.IMREAD_UNCHANGED)
	im4 = cv2.imread('./test-im/frrn/4-5.png', cv2.IMREAD_UNCHANGED)
	patches.append(im1[256:,256:])
	patches.append(im2[256:,:256])
	patches.append(im3[:256,256:])
	patches.append(im4[:256,:256])
	mask = bilinear_warp(patches, orit='V')
	#print mask[0][255,0]
	#im_color = cv2.applyColorMap((mask*255).astype('uint8'),
	#	cv2.COLORMAP_JET)
	cv2.imshow('0', mask)
	cv2.waitKey(0)
	'''
	