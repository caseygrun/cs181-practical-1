# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# pca.py
# Does principal component analysis and singular value decomposition
# 
# ----------------------------------------------------------------------------


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

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
	Projects the sparse matrix X on to the basis V.

	X is (N x D) and V is (K x D)
	"""
	pass
