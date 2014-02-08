# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun, Rhed Shi, Sam Kim
# 
# visualize.py
# Contains functions for visualizing CIFAR-10 data and sparsity
# 
# ----------------------------------------------------------------------------


import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def imshape(a,n,m):
	"""
	Reshapes an array into an (N x M x 3) array
	"""
	b = a.reshape((n,m,3),order='F').swapaxes(0,1)
	return b

def im_show_grid(U):
	"""
	Shows a grid of CIFAR-10 images or CIFAR-10 image means. U should be an 
	(N x 3072) matrix of N CIFAR-10 images (or image means).
	"""
	fig = plt.figure(1, (4., 4.))
	width = 5
	grid = ImageGrid( fig, 111, nrows_ncols=(int(math.ceil(U.shape[0]/width)),width) )

	for i in range(U.shape[0]):
		grid[i].imshow(imshape(U[i,:],32,32))

	plt.show()

def show_sparsity(R):
	"""
	Shows a plot showing sparsity 
	"""
	plt.spy(R, precision=1e-3, marker='.', markersize=3)
	plt.show()