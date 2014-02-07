# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Warmup
# Casey Grun
# 
# kmeans.py
# Implements k-means clustering for an arbitrary data set, given a distance
# function.
# ----------------------------------------------------------------------------

import numpy as np


# display status information
DEBUG = True

def debug(fmt, arg=tuple()):
	if DEBUG: print fmt % arg

def unpickle(file):
	"""
	Loads one of the CIFAR-10 files
	"""
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def pickle(array, file):
	"""
	Dumps an array to a file
	"""
	import cPickle
	fo = open(file,'wb')
	cPickle.dump(array,fo)
	fo.close()


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
	Rotates a given solution
	"""
	U = np.roll(U,1,axis=0)
	R = np.roll(R,1,axis=1)
	return (U,R)




# inverse transform sampling
# http://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability
import random
import bisect
import collections

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
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    return (data - mean)/std


# k-means clustering
def cluster(X, K, distance):
	"""
	Performs K-means clustering on the data set X, using the provided distance
	function.

	:param X: an N x D array of data points
	:param K: the number of clusters
	:param distance: a distance function; should accept X [N x D] and U [K x D]
		and return an array [K x N] containing the distances between each row 
		in X and each row in U. 

	:return: (U, R) where U is [K x D] containing the cluster centers, and 
		R is [N x K] containing the cluster responsibilities

	"""

	# number of data points
	N = X.shape[0]

	# dimensionality of data points
	D = X.shape[1]

	# cluster means
	U = np.zeros((K,D))

	# empty responsibilities
	R = np.zeros((N,K)) 

	debug("Initializing...")

	# initialize responsibilities with k-means++
	# pick random data point for initial cluster
	n = random.randrange(N)
	print n
	U[0,:] = X[n,:]

	# loop over remaining clusters
	for k in range(1,K):
		# compute distance between each data point and all clusters so far
		# D is (k x N)
		L = distance(X,U[0:k,:])

		# pick minimum squared distance for each data point
		d = np.min(L, axis=0)**2

		# calculate a probability distribution according to distance---more 
		# distant data should have higher probability
		p = d / sum(d)

		# choose the next cluster by sampling from this distribution
		i = choice(range(N),p)
		U[k,:] = X[i,:]

	if DEBUG:
		print "Clustering..."

	# repeat...
	while True:

		# set responsibilities
		
		# copy responsibilities
		Rp = R

		# zero the responsibilities
		R = np.zeros((N,K))

		# compute distances between each data point and each mean
		L = distance(X,U) 

		# find the index of the nearest cluster, for each data point 
		# (argmin over the K dimension, so kp is (N x 1))
		kp = np.argmin(L,axis=0)

		# assign responsibilities
		for n in range(N):
			R[n,kp[n]] = 1

		# ...break if the R's don't change
		if np.all(R == Rp): 
			break
		
		if DEBUG:
			print sum(Rp - R)


		# determine cluster centers

		# figure out how many points are in each cluster
		# Nk is (K x 1)
		Nk = np.sum(R,axis=0)
		
		# calculate cluster centers
		# R is (N x K), X is (N x D) -> dot(R.transpose(),X) is (K x D) 
		# represents the means
		U = np.dot(R.transpose(),X) / Nk[:,np.newaxis]
			
	return (U, R)

# k-means clustering for sparse arrays
def cluster2(X, K, distance):
	"""
	Performs K-means clustering on the data set X, using the provided distance
	function.

	:param X: an N x D array of data points
	:param K: the number of clusters
	:param distance: a distance function; should accept X [N x D] and U [K x D]
		and return an array [K x N] containing the distances between each row 
		in X and each row in U. 

	:return: (U, R) where U is [K x D] containing the cluster centers, and 
		R is [N x K] containing the cluster responsibilities

	"""

	# number of data points
	N = X.shape[0]

	# dimensionality of data points
	D = X.shape[1]

	# cluster means
	U = np.zeros((K,D))

	# empty responsibilities
	R = np.zeros((N,K)) 

	if DEBUG:
		print "Initializing..."

	# initialize responsibilities with k-means++
	# pick random data point for initial cluster
	n = random.randrange(N)
	U[0,:] = X[n,:].toarray()

	# loop over remaining clusters
	for k in range(1,K):
		debug("%d / %d",(k,K))

		# compute distance between each data point and all clusters so far
		# D is (k x N)
		L = distance(X,U[0:k,:])

		# pick minimum squared distance for each data point
		d = np.min(L, axis=0)**2

		# calculate a probability distribution according to distance---more 
		# distant data should have higher probability
		p = d / sum(d)

		# choose the next cluster by sampling from this distribution
		i = choice(range(N),p)
		U[k,:] = X[i,:].toarray()

	if DEBUG:
		print "Clustering..."

	# repeat...
	while True:

		# set responsibilities
		
		# copy responsibilities
		Rp = R

		# zero the responsibilities
		R = np.zeros((N,K))

		# compute distances between each data point and each mean
		L = distance(X,U) 

		# find the index of the nearest cluster, for each data point 
		# (argmin over the K dimension, so kp is (N x 1))
		kp = np.argmin(L,axis=0)

		# assign responsibilities
		for n in range(N):
			R[n,kp[n]] = 1

		# ...break if the R's don't change
		if np.all(R == Rp): 
			break
		
		if DEBUG:
			print sum(Rp - R)


		# determine cluster centers

		# figure out how many points are in each cluster
		# Nk is (K x 1)
		Nk = np.sum(R,axis=0)
		
		# calculate cluster centers
		# R is (N x K), X is (N x D) -> dot(R.transpose(),X) is (K x D) 
		# and represents the means

		# non-broadcasting version
		for k in range(K):
			U[k,:] = X.transpose().dot(R[:,k]) / Nk[k]
			
	return (U, R)