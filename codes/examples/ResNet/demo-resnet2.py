import numpy as np
import cv2, os

#from linematch_resnet import *
#from geomatric_transform import cos_angle_h, rotate_full, affine

def cos_angle_h(ln, h_ln=(1.,0.)):
	v = line_vec(ln)
	arc = np.arccos(v[0]*h_ln[0] + v[1]*h_ln[1])
	if v[1] > 0:
		arc = -arc
	return arc

def rotate_full(im_shape, theta):
	# center
	(h, w) = im_shape
	(cX, cY) = (w // 2, h // 2)
	angle = theta*180.0/np.pi
	M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))
	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY
	return M, (nW, nH)

def affine(point, trans_matrix):
	x, y = point
	ax = trans_matrix[0,0]*x + trans_matrix[0,1]*y + trans_matrix[0,2]
	ay = trans_matrix[1,0]*x + trans_matrix[1,1]*y + trans_matrix[1,2]
	return [int(ax+0.5), int(ay+0.5)]

def impari_load(im_path1, im_path2):
	left_im = cv2.imread(im_path1, cv2.IMREAD_COLOR)
	right_im = cv2.imread(im_path2, cv2.IMREAD_COLOR)
	return left_im, right_im

def line_length(line):
	return np.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)

def line_center(line):
	return [(line[0]+line[2])/2.0, (line[1]+line[3])/2.0]

def line_load(data_path):
	fp = open(data_path).read().splitlines()
	lines = []
	for l in fp:
		line = l.split(' ')[1:]
		for i in range(8):
			line[i] = int(float(line[i]))
		lines.append(line)
	return lines

def line_read(data_path):
	fp = open(data_path).read().splitlines()
	lines = []
	for l in fp:
		line = l.split(' ')
		idx = 0
		ln = []
		for i in range(len(line)):
			if line[i] != '':
				ln.append(int(float(line[i]) + 0.5))
		lines.append(ln)
	return lines

def adjust_center(center, w, height, width):
	#print center, height, width
	if center[0] < w: 
		center[0] = w
	if center[0] > width-w: 
		center[0] = width-w
	if center[1] < w: 
		center[1] = w
	if center[1] > height-w: 
		center[1] = height-w
	center[0] = int(center[0])
	center[1] = int(center[1])

def triplets(model_path, im_path1, line_file1, im_path2, line_file2, output):
	lim, rim = impari_load(im_path1, im_path2)
	llines = line_read(line_file1)
	rlines = line_read(line_file2)

	m, n = len(llines), len(rlines)

	width, r = 128, 64
	szl, szr = lim.shape, rim.shape

	ims_left = np.zeros((m, r, width, 4), dtype=np.uint8)
	ims_right = np.zeros((n, r, width, 4), dtype=np.uint8)

	for i in range(m):
		ln = llines[i]
		theta = cos_angle_h(ln)
		M, nsz = rotate_full(sz[:2], theta)

		#r_mat = rotate_matrix((sz[1]/2., sz[0]/2.), cos_angle_h(ln))
		p1 = affine([ln[0], ln[1]], M)
		p2 = affine([ln[2], ln[3]], M)
		left_mask = np.zeros((nsz[1], nsz[0]),dtype=np.uint8)
		cv2.line(left_mask, (p1[0], p1[1]), (p2[0], p2[1]), color=255, thickness=1)
		#cent1 = line_center(llines[i])
		#adjust_center(cent1, r, szl[0], szl[1])
		im_rotate = cv2.warpAffine(lim, M, nsz)

		lx = max(min(p1[0]-32, p2[0]-32),0)
		ly = max(min(p1[1]-32, p2[1]-32),0)
		rx = min(max(p1[0]+32, p2[0]+32),nsz[0])
		ry = min(max(p1[1]+32, p2[1]+32),nsz[1])
		ims_left[i,:,:,:3] = im_rotate[ly:ry, lx:rx,:]
		ims_left[i,:,:,3] = left_mask[ly:ry, lx:rx,:]

	for i in range(n):
		ln = rlines[i]
		theta = cos_angle_h(ln)
		M, nsz = rotate_full(sz[:2], theta)

		#r_mat = rotate_matrix((sz[1]/2., sz[0]/2.), cos_angle_h(ln))
		p1 = affine([ln[0], ln[1]], M)
		p2 = affine([ln[2], ln[3]], M)
		right_mask = np.zeros((nsz[1], nsz[0]),dtype=np.uint8)
		cv2.line(right_mask, (p1[0], p1[1]), (p2[0], p2[1]), color=255, thickness=1)
		#cent1 = line_center(llines[i])
		#adjust_center(cent1, r, szl[0], szl[1])
		im_rotate = cv2.warpAffine(rim, M, nsz)
		lx = max(min(p1[0]-32, p2[0]-32),0)
		ly = max(min(p1[1]-32, p2[1]-32),0)
		rx = min(max(p1[0]+32, p2[0]+32),nsz[0])
		ry = min(max(p1[1]+32, p2[1]+32),nsz[1])
		ims_right[i,:,:,:3] = im_rotate[ly:ry, lx:rx,:]
		ims_right[i,:,:,3] = right_mask[ly:ry, lx:rx,:]

	image = np.zeros((width, width, 4), dtype=np.uint8)
	image[:r,:,:] = ims_left[i]
	image[r:,:,:] = ims_right[j]

	im2 = image[:,:,:3]
	im2 = im2[:,:,2]*0.6 + image[:,:,3]*0.4
	cv2.imshow('0', im2.astype('uint8'))
	cv2.waitKey(0)
	'''
	corr = np.zeros((m,n))
	pred_config = PredictConfig(
		model=Model(),
		session_init=get_model_loader(model_path),
		input_names=['input'],
		output_names=['output']
	)
	predictor = OfflinePredictor(pred_config)

	ret = []
	for i in range(m):
		for j in range(n):
			image = np.zeros((width, width*2, 4), dtype=np.uint8)
			image[:,:width,:] = ims_left[i]
			image[:,width:,:] = ims_right[j]

			image = image.astype('float32')
			image = image[np.newaxis, :, :, :]

			prob = predictor([image])[0][0]
			corr[i,j] = prob[1]

	corr = corr*255
	corr = corr.astype('uint8')
	cv2.imwrite(os.path.join(output, 'corrmat.png'), corr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
	'''


if __name__ == '__main__':
	model_path = '/home/swoda/tensorpack/examples/train_log/linematch-resnet/model-66000.data-00000-of-00001'
	im1_path = '/home/swoda/Downloads/LineMatchingBenchmark-master/benchmark/building_rotation/1.jpg'
	im2_path = '/home/swoda/Downloads/LineMatchingBenchmark-master/benchmark/building_rotation/2.jpg'
	line_file1 = '/home/swoda/Downloads/LineMatchingBenchmark-master/benchmark/building_rotation/ed1.txt'
	line_file2 = '/home/swoda/Downloads/LineMatchingBenchmark-master/benchmark/building_rotation/ed2.txt'
	output = '/home/swoda/tensorpack/examples/LineMatch'
	triplets(model_path, im1_path, line_file1, im2_path, line_file2, output)
