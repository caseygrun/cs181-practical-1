# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Warmup
# Casey Grun
# 
# warmup.py
# Calls the k-means clustering algorithm defined in kmeans.py
# ----------------------------------------------------------------------------

import numpy as np
import kmeans

def dist(x,u):
	"""
	Given an (n x d) array X, a (k x d) array U, returns a (k x n) array D 
	giving the cartesian distances between each row of X and each row of U. 
	That is, D[i,j] = distance(U[i,:], X[j,:])
	"""
	# https://github.com/dwf/rescued-scipy-wiki/blob/master/EricsBroadcastingDoc.rst
	diff = x[np.newaxis,:,:] - u[:,np.newaxis,:]
	dist = np.sum(diff**2,axis=-1)
	return dist

def roll(U,R):
	"""
	Rotates a given solution
	"""
	U = np.roll(U,1,axis=0)
	R = np.roll(R,1,axis=1)
	return (U,R)


# Try a simple example that's easily verified
X = np.array([[10, 0, 0, 0],
			  [11, 0, 0, 0],
			  [9 , 0, 0, 0],
			  [0 , 8, 0, 0],
			  [0 , 6, 0, 0],
			  [0 , 7, 0, 0],
			  [0 , 0, 9, 0],
			  [0 , 0, 8, 0],
			  [0 , 0,10, 0],])
(U, R) = kmeans.cluster(X,3,dist)

print U
print R

