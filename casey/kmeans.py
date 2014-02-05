# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Warmup
# Casey Grun
# 
# kmeans.py
# Implements k-means clustering for an arbitrary data set, given a distance
# function.
# ----------------------------------------------------------------------------

import numpy as np


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


	# initialize responsibilities with k-means++
	# pick random data point for initial cluster
	n = random.randrange(N)
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
		if np.all(R == Rp): break


		# determine cluster centers

		# figure out how many points are in each cluster
		# Nk is (K x 1)
		Nk = np.sum(R,axis=0)
		
		# calculate 
		# R is (N x K), X is (N x D) -> dot(R.transpose(),X) is (K x D) 
		# and represents the means
		U = np.dot(R.transpose(),X) / Nk[:,np.newaxis]
			
	return (U, R)