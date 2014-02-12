# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# mf.py
# Does matrix factorization
# ----------------------------------------------------------------------------


import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg
import shared_utils as su
import time
import math

DEBUG = True
def debug(fmt, arg=tuple()):
	if DEBUG: print fmt % arg

# steps=5000, alpha=0.0002, beta=0.02, epsilon=0.001, save_every=20
def mfact2(R, N, D, K, steps=500, alpha=0.01, beta=0.02, epsilon=0.001, save_every=20, use_bias=True, filename=None):
	"""
	Adapted from Albert Au Yeung (2010)
	http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

	Arguments:
		R       	: the set of known ratings, given as a list of tuples 
					(i, j, r) where i is the index of the user, j is the index 
					of the book, and r is the rating
		K       	: the number of latent features
		steps   	: the maximum number of steps (epochs) to perform the optimisation
		alpha   	: the learning rate
		beta    	: the regularization parameter
		epsilon 	: the minimum difference in error (below which to quit)
		save_every	: save results after every `save_every` iterations
		filename	: file to save results to. Defaults to "output/mfact_K_step"

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

	if(not use_bias):
		Bn = np.zeros((N,1))
		Bd = np.zeros((D,1))

	# initialize total error
	e = 0
	ep = 0

	# small function to serialize the results
	def results():
		return {"P":P, "Q":Q, "Bn":Bn, "Bd":Bd, "mean":mean, "step":step, "de":de, "e":e}

	# calculate mean of R
	debug("Calculating mean of R...")
	mean = 1./len(R) * float(sum([ Rij for (i,j,Rij) in R ]))
	debug("Mean: %f", (mean,))

	t = time.clock()
	debug("Starting Matrix Factorization into %d principal components...", (K,))
	debug("Step \t Of \t Time \t Error \t Change")

	# set default filename
	if(filename==None):
		filename = "output/mfact_%d" % K

	# for each step (epoch)
	for step in xrange(steps):
		
		# remember the previous error
		ep = e
		
		# calculate the total error, e
		e = 0
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			e += pow(eij, 2)

		e = e + (beta/2) * (sum(Bn**2) + sum(Bd**2) + la.norm(P) + la.norm(Q))
		de = ep - e

		# report error and timing
		tp = time.clock()
		debug("%d \t %d \t %d \t %d \t %d", (step, steps, tp-t, e, de))
		t = tp

		# break if error is small enough
		if abs(de) < epsilon:
			print "Finished after step %d with delta-error = %f" % (step, de)
			break

		# periodically save results
		if (step % save_every) == 0:
			su.pickle(results(),filename + "_%d" % step)

		dP = np.zeros_like(P)
		dQ = np.zeros_like(Q)
		dBn = np.zeros_like(Bn)
		dBd = np.zeros_like(Bd)
		
		# calculate gradient
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			# P[i,:] = P[i,:] + alpha * (2 * eij * Q[j,:] - beta * P[i,:])
			# Q[j,:] = Q[j,:] + alpha * (2 * eij * P[i,:] - beta * Q[j,:])
			# Bn[i]  = Bn[i]  + alpha * (2 * eij          - beta * Bn[i]) 
			# Bd[j]  = Bd[j]  + alpha * (2 * eij          - beta * Bd[j]) 
			dP[i,:] += alpha * (2 * eij * Q[j,:] - beta * P[i,:])
			dQ[j,:] += alpha * (2 * eij * P[i,:] - beta * Q[j,:])
			if use_bias:
				dBn[i]  += alpha * (2 * eij          - beta * Bn[i]) 
				dBd[j]  += alpha * (2 * eij          - beta * Bd[j]) 

		# update P, Q, Bn, Bd
		P += dP
		Q += dQ
		Bn += dBn
		Bd += dBd

	if step == steps-1:
		print "Gave up after %d steps with delta-error = %f" % (step+1, de)

	# return results
	return results()

mfact = mfact2