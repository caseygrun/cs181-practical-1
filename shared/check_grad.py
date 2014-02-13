# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# check_grad.py
# Tries to check the gradient with a finite-difference approximation
# ----------------------------------------------------------------------------

import numpy as np
import numpy.linalg as la
import shared_utils as su
import random
import math
import sys
import scipy.optimize

def check(N, D, K, alpha=0.01, beta=0.02, epsilon=1e-4):

	def unroll(A):
		"""
		Takes a big matrix A and splits it up into P, Q, Bn, and Bd
		"""
		P  = A[0             : N*K ]             .copy().reshape(N,K)
		Q  = A[N*K           : N*K + D*K ]       .copy().reshape(D,K)
		Bn = A[N*K + D*K     : N*K + D*K + N ]   .copy().reshape(N,1)
		Bd = A[N*K + D*K + N : N*K + D*K + N + D].copy().reshape(D,1)
		return (P,Q,Bn,Bd)

	def compact(P,Q,Bn,Bd):
		"""
		Takes four matrices (P, Q, Bn, and Bd) and concatenates them into one
		flat matrix A
		"""
		return np.concatenate((P.flat, Q.flat, Bn.flat, Bd.flat)).copy()

	def error(A,R):
		"""
		Calculates the errors
		"""
		(P,Q,Bn,Bd) = unroll(A.copy())

		# calculate the total error, e
		e = 0
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			e += eij**2

			# e += (beta/2) * (Bn[i]**2 + Bd[j]**2)
			# e += (beta/2) * (np.sum(P[i,:]**2) + np.sum(Q[j,:]**2))

		# e = e + (beta/2) * (sum(Bn**2) + sum(Bd**2) + la.norm(P) + la.norm(Q))
		e = e + (beta/2) * (sum(Bn**2) + sum(Bd**2) + np.sum(P**2) + np.sum(Q**2))

		return e

	def grad(A,R):
		(P,Q,Bn,Bd) = unroll(A.copy())

		dP = np.zeros_like(P)
		dQ = np.zeros_like(Q)
		dBn = np.zeros_like(Bn)
		dBd = np.zeros_like(Bd)

		# calculate gradient
		for (i,j,Rij) in R:
			eij = Rij - (mean + Bn[i] + Bd[j] + np.dot(P[i,:],Q[j,:]))
			dP[i,:] += -1. * (2 * eij * Q[j,:] - beta * P[i,:])
			dQ[j,:] += -1. * (2 * eij * P[i,:] - beta * Q[j,:])
			dBn[i]  += -1. * (2 * eij          - beta * Bn[i]) 
			dBd[j]  += -1. * (2 * eij          - beta * Bd[j]) 
			# dBn[i] = 0.
			# dBd[j] = 0.


		# return gradient
		return compact(dP, dQ, dBn, dBd)

	# scipy's check_grad function only lets us use a flat, 1-D array, so I 
	# concatenate all our input coordinates together into one long flat 1-D 
	# array.
	
	# pick a random place in coordinate space to start out
	A = np.random.random_sample((N*K + D*K + N +D,))
	# A[N*K + D*K:] = 0
	assert (A == compact(*unroll(A))).all()

	# generate some random data for R
	samples = max(int(math.ceil(N*D/2)),1)
	chosen = set()
	R = []
	for s in xrange(samples):
		i = random.randrange(N)
		j = random.randrange(D)
		Rij = random.uniform(-1,1)
		if (i,j) not in chosen:
			R += [(i, j, Rij)]	
			chosen.add((i,j))

	# # uncomment this to set values for all i,j in R
	# R = [(i,j,random.uniform(0,1)) for i in xrange(N) for j in xrange(D)]
	
	# calculate mean of values in R
	mean = 1./len(R) * float(sum([ Rij for (i,j,Rij) in R ]))

	def check_grad(func,grad,x0,*args):
		g = grad(x0, *args)
		f = scipy.optimize.approx_fprime(x0, func, epsilon, *args)
		err = np.sqrt(sum((g - f)**2)/len(x0))
		print "G: ", 
		print g
		print "F: ",
		print f
		print "G - F: ",
		print g - f
		print "RMSE: ",
		print err
		return err

	return check_grad(error,grad,A,R)
	# return scipy.optimize.check_grad(error,grad,A,R)


# print check(50,50,1)

# check 10 times and average
# enter different values for N, D, and K here. 
trials = 1
rmses = [check(50,50,5) for x in xrange(trials)]
print "Average RMSE over %d trials: " % trials,
print np.mean(rmses)
sys.exit(0)

# ------------------------------------------------------------

# check different values for N, D, K
values = [1, 5, 10, 25, 50]
rmses = np.zeros((len(values),len(values),len(values)))
for (i,N) in enumerate(values):
	for (j,D) in enumerate(values):
		for (k,K) in enumerate(values):
			rmses[i,j,k] = check(N,D,K)
			print "N = %d \t D = %d \t K = %d \t RMSE = %d" % (N,D,K,rmses[i,j,k])

su.pickle((values,rmses),"check_grad_data_b_0.02")
sys.exit(0)

# ------------------------------------------------------------

# plot gradient RMSE
(values,rmses) = su.unpickle("check_grad_data")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for (i,N) in enumerate(values):
	for (j,D) in enumerate(values):
		print (i,j), (N,D)
		plt.subplot(len(values),len(values),j*len(values)+i+1)
		plt.plot(values,rmses[i,j,:])
		plt.title("N = %d, D = %d" % (N, D))
		plt.xlabel("K")
		plt.ylabel("RMSE")
		plt.ylim(0,100)

plt.show()

