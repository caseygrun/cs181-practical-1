# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# pca.py
# Does principal component analysis and singular value decomposition
# 
# ----------------------------------------------------------------------------


import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg
import shared_utils as su
import time

DEBUG = True

def debug(fmt, arg=tuple()):
	if DEBUG: print fmt % arg



def cov(X):
	"""
	Calculates the covariance matrix of X, if X is (N x D)
	"""
	N = X.shape[0]
	D = X.shape[1]
	S = np.zeros((D,D))
	Xbar = X.mean(0)
	for n in xrange(N):
		Xn = X[n,:].toarray() - Xbar
		S += 1./N * np.dot(Xn.transpose(),Xn)
	return S



def pca(X,K):
	"""
	Does principal components analysis on the sparse matrix X
	"""
	debug("Calculating covariance matrix...")
	S = cov(X)

	debug("Computing eigenspectrum...")
	(w, v) = scipy.sparse.linalg.eigs(S, k=K, which='LM')

	debug("Done.")
	return (w, v)

analyze = pca

def svd(X,K):
	"""
	Does singular value decomposition on the sparse matrix X.

	Accepts X (M x N) and calculates the first K singular values and singular
	vectors by singular value decomposition. 

	Returns: 
	u : ndarray, shape=(M, k)
		Unitary matrix having left singular vectors as columns.
	s : ndarray, shape=(k,)
		The singular values.
	vt : ndarray, shape=(k, N)
		Unitary matrix having right singular vectors as rows.
	"""
	debug("Calculating %d singular values by SVD...",(K,))
	(u, s, vt) = scipy.sparse.linalg.svds(X, k=K, which='LM')
	return (u, s, vt)

def project(X,V):
	"""
	Projects the array X on to the basis V.

	X is (N x D) and V is (K x D) 
	-> X' = X . V^T is (N x K)
	"""
	return X.dot(V.transpose())

def project2(X,V):
	"""
	Projects the sparse matrix X on to the basis V (non-broadcasting version 
	of `project`).

	X is (N x D) and V is (K x D) 
	-> X' = X . V^T is (N x K)
	"""
	Vt = V.transpose()
	N = X.shape[0]
	K = V.shape[0]
	Xp = np.zeros((N,K))
	for n in xrange(N):
		# X[n,:] is (1 x D), V^T is (D x K) so X[n,:] . V^T is (1 x K)
		Xp[n,:] = X[n,:].toarray().dot(Vt)
	return Xp



def mfact(R, K, steps=5000, alpha=0.0002, beta=0.02):
	"""
	Adapted from Albert Au Yeung (2010)
	http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

	Arguments:
		R     : a matrix to be factorized, dimension N x D    
		K     : the number of latent features
		steps : the maximum number of steps to perform the optimisation
		alpha : the learning rate
		beta  : the regularization parameter

	Returns:
		P     : an initial matrix of dimension N x K
		Q     : an initial matrix of dimension D x K
	"""
	N = R.shape[0]
	D = R.shape[1]

	P = np.random.rand(N,K)
	Q = np.random.rand(D,K)
	Q = Q.T

	# Biases
	Bu = np.zeros((N,1)) # N x 1
	Bi = np.zeros((D,1)) # D x 1

	# mean = ?? # calculate mean of R


	debug("Starting Matrix Factorization into %d principal components...", (K,))

	for step in xrange(steps):
		for i in xrange(N):
			for j in xrange(D):
				if R[i,j] > 0:
					eij = R[i,j] - np.dot(P[i,:],Q[:,j])
					for k in xrange(K):
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
		eR = np.dot(P,Q)
		e = 0
		for i in xrange(N):
			for j in xrange(D):
				if R[i,j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
					for k in xrange(K):
						e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

		debug("Step %d / %d: e = %d", (step, steps,e))
		if e < 0.001:
			break
	return P, Q.T

# steps=5000, alpha=0.0002, beta=0.02, epsilon=0.001, save_every=20
def mfact2(R, N, D, K, steps=5000, alpha=0.01, beta=0.02, epsilon=0.001, save_every=20):
	"""
	Adapted from Albert Au Yeung (2010)
	http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

	Arguments:
		R       	: the set of known ratings, given as a list of tuples 
					(i, j, r) where i is the index of the user, j is the index 
					of the book, and r is the rating
		K       	: the number of latent features
		steps   	: the maximum number of steps to perform the optimisation
		alpha   	: the learning rate
		beta    	: the regularization parameter
		epsilon 	: the minimum error (below which to quit)
		save_every	: save results after every `save_every` iterations

	Returns: a dict with the following keys:
		"P"     	: an array (N x K) containing user features
		"Q"     	: an array (D x K) containing book features
		"Bn"    	: an array (N x 1) containing user biases
		"Bd"    	: an array (D x 1) containing book biases
		"mean"  	: the global mean of the initial rankings
	"""
	
	# initialize random user and book feature matrices
	P = np.random.rand(N,K)
	Q = np.random.rand(D,K)

	# initialize random bias vectors
	Bn = np.random.rand(N,1) # N x 1
	Bd = np.random.rand(D,1) # D x 1

	# Error
	# E = np.empty((N,D))

	# calculate mean of R
	debug("Calculating mean of R...")
	mean = 1./len(R) * float(sum([ Rij for (i,j,Rij) in R ]))
	debug("Mean: %f", (mean,))

	t = time.clock()
	debug("Starting Matrix Factorization into %d principal components...", (K,))

	# for each step (epoch)
	for step in xrange(steps):
		
		# calculate the total error, e
		e = 0
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			e += pow(eij, 2)

		e = e + (beta/2) * (sum(Bn**2) + sum(Bd**2)) + la.norm(P) + la.norm(Q)

		# report error and timing
		tp = time.clock()
		debug("Step %000d / %000d (%d): e = %d", (step, steps, tp-t, e))
		t = tp

		# break if error is small enough
		if e < epsilon:
			break

		# periodically save results
		if (step % save_every) == 0:
			su.pickle({"P":P, "Q":Q, "Bn":Bn, "Bd":Bd, "mean":mean},"output/mfact_%d_%d" % (K,step))

		# update P, Q, Bn, and Bd by gradient descent
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			# eij = E[i,j]
			P[i,:] = P[i,:] + alpha * (2 * eij * Q[j,:] - beta * P[i,:])
			Q[j,:] = Q[j,:] + alpha * (2 * eij * P[i,:] - beta * Q[j,:])
			Bn[i]  = Bn[i]  + alpha * (2 * eij          - beta * Bn[i]) 
			Bd[j]  = Bd[j]  + alpha * (2 * eij          - beta * Bd[j]) 


	# return results
	return {"P":P, "Q":Q, "Bn":Bn, "Bd":Bd, "mean":mean}

