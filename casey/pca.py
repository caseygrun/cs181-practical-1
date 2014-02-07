import numpy as np
import scipy.sparse as sp


def cov(X):
	"""
	X is (N x D)
	"""
	N = X.shape[0]
	D = X.shape[1]
	S = np.zeros(D,D)
	Xbar = X.mean(0)
	for n in xrange(N)
		Xn = X[n,:].toarray() - Xbar
		S += 1/N * (Xn * Xn.transpose())
	return S


def pca(X,K):
	S = cov(X)

	(w, v) = sp.linalg.eigs(S, k=K, which='LM')
	return (w, v)