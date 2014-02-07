import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

DEBUG = True

def debug(fmt, arg=tuple()):
	if DEBUG: print fmt % arg


def cov(X):
	"""
	X is (N x D)
	"""
	N = X.shape[0]
	D = X.shape[1]
	S = np.zeros((D,D))
	Xbar = X.mean(0)
	for n in xrange(N):
		Xn = X[n,:].toarray() - Xbar
		S += 1./N * np.dot(Xn.transpose(),Xn)
	return S


def analyze(X,K):
	debug("Calculating covariance matrix...")
	S = cov(X)

	debug("Computing eigenspectrum...")
	(w, v) = scipy.sparse.linalg.eigs(S, k=K, which='LM')

	debug("Done.")
	return (w, v)