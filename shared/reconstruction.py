# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# reconstruction.py
# Picks a random P, Q and produces an R from that factorization. Tries to get
# mfact to reconstitute P, Q
# ----------------------------------------------------------------------------

import sys
import math
import random
import numpy as np
import books
import visualize
import mf
import util
import shared_utils as su



# choose a size of P and Q
N = 100
D = 100

# choose a number of features, limit the time the simulation runs
K = 3

# update parameters
alpha=0.001
beta=0.02
epsilon=0.00005
max_steps = 2000 # change this to something reasonable, like 200 or 500

withhold = 0.

# update this for each trial you do with a particular k
run = 2

# generate random P, Q
P = np.random.random_sample((N,K))
Q = np.random.random_sample((D,K))

# generate faux R based on real P and Q
R = np.dot(P,Q.T)
Ra = [(i, j, R[i,j]) for i in xrange(N) for j in xrange(D) ]

# withhold a random portion of the data
random.shuffle(Ra)
Ra = Ra[0:max(int(len(Ra)*(1-withhold)),1)]


# do factorization
data_mfact = mf.mfact(Ra, N, D, \
	K, steps=max_steps, alpha=alpha, beta=beta, epsilon=epsilon, \
	use_bias=False, filename=("output/reconstruct/mfact_%d_run_%d" % (K, run)))

# cross-validate with the original P, Q
Pp = data_mfact["P"]
Qp = data_mfact["Q"]
Rp = np.dot(Pp,Qp.T) + data_mfact["mean"]

print R - Rp

rmse = np.sqrt(np.sum((R-Rp)**2))
print_table()

def print_table():
	print "       N	       D	       K	   alpha	    beta	     eps	   steps	    RMSE"
	print "%8d	%8d	%8d	%8f	%8f	%8f	%8d	%8f" % (N, D, K, alpha, beta, epsilon, data_mfact["step"], rmse)
