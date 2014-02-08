
# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun, Rhed Shi, Sam Kim
# 
# shared_utils.py
# Contains shared utility functions for unsupervised learning
# 
# to link this file into your directory, just do this (from the /shared 
# directory):
# 
# ln -s shared_utils.py ../YOURNAME/shared_utils.py
# 
# ----------------------------------------------------------------------------

import numpy as np
import random
import bisect
import collections

# load and save pickled data structures
def unpickle(file):
	"""
	Loads and returns a pickled data structure in the given `file` name

	Example usage:

		data = unpickle('output/U_20_std')

	"""
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def pickle(array, file):
	"""
	Dumps an array to a file. 

	Example usage:

		pickle(U, 'output/U_20_std')
		
	"""
	import cPickle
	fo = open(file,'wb')
	cPickle.dump(array,fo)
	fo.close()

# calculate cartesian distances
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

def dist2(x,u):
	"""
	Given an (n x d) array X, a (k x d) array U, returns a (k x n) array D 
	giving the cartesian distances between each row of X and each row of U. 
	That is, D[i,j] = distance(U[i,:], X[j,:]). This function does not use
	broadcasing.
	"""
	N = x.shape[0]; D = x.shape[1]; K = u.shape[0] 
	dist = np.zeros((K,N))
	for i in xrange(N):
		diff = u - x[i,:].toarray()
		dist[:,i] = np.sum(diff**2,axis=-1)
	return dist


def roll(U,R):
	"""
	Rotates a given K-means solution
	"""
	U = np.roll(U,1,axis=0)
	R = np.roll(R,1,axis=1)
	return (U,R)




# inverse transform sampling
# http://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability

def cdf(weights):
	"""
	Determine empirical CDF of a set of weights
	"""
	return np.cumsum(weights) / sum(weights)

def choice(population,weights):
	"""
	Choose from a population with the corresponding weights. Uses inverse 
	transform sampling (universality of the uniform) to sample from the
	empirical CDF.
	"""
	assert len(population) == len(weights)
	cdf_vals=cdf(weights)
	return population[bisect.bisect(cdf_vals, random.random())]


# data standardization
def standardize(data):
    """
    Take an NxD numpy matrix as input and return a standardized version of it.
    """
    mean = data.mean(axis=0)
    std  = data.std(axis=0)
    return (data - mean)/std
