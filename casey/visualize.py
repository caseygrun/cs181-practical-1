import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def imshape(a,n,m):
	b = a.reshape((n,m,3),order='F').swapaxes(0,1)
	return b

def im_show_grid(U):

	fig = plt.figure(1, (4., 4.))
	width = 5
	grid = ImageGrid( fig, 111, nrows_ncols=(int(math.ceil(U.shape[0]/width)),width) )

	for i in range(U.shape[0]):
		grid[i].imshow(imshape(U[i,:],32,32))

	plt.show()

