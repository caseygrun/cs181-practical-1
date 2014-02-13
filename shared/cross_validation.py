# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# cross-validation.py
# Does cross-validation with a set of withheld data
# ----------------------------------------------------------------------------

import sys
import math
import numpy as np
import books
import visualize
import mf
import util
import shared_utils as su
import signal


def print_table():
	print "       N	       D	       K	   alpha	    beta	     eps	   steps	  points	  w/held	 discard	    RMSE"
	print "%8d	%8d	%8d	%8f	%8f	%8f	%8d	%8d	%8d	%8d	%8f" % (N, D, K, alpha, beta, epsilon, data_mfact["step"], T, withhold, discard, rmse)

# def signal_handler(signal, frame):
#     print_table()
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)

# choose a number of features, limit the time the simulation runs
K = 5

# update parameters
alpha=0.001
beta=0.02
epsilon=0.00005
max_steps = 1000 # change this to something reasonable, like 200 or 500

T = 200000
withhold = 2000
discard = 0 #T-20000

# update this for each trial you do with a particular k
run = 3

# do this once to build the ratings and save them to ratings_tuple_std
books.build_ratings(filename="ratings_tuple_std", standardize=True, withhold=withhold, discard=discard)

# load training data
data_train = su.unpickle("ratings_tuple_std")

N = data_train["N"]
D = data_train["D"]

data_mfact = mf.mfact(data_train["ratings"], data_train["N"], data_train["D"], \
	K, steps=max_steps, alpha=alpha, beta=beta, epsilon=epsilon, \
	filename=("output/large/mfact_%d_run_%d" % (K, run)))

# cross-validate with the withheld data
data_withheld = su.unpickle("ratings_tuple_std_withheld")
rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)

# print results of cross-validation
print_table()

# # make some predictions
predictions = books.make_predictions(data_train, data_mfact)

# # write the predictions
util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))

# ----------------------------------------------------------------------------

# # cross-validate offline

# # load training data
# data_train = su.unpickle("ratings_tuple_std")

# # cross-validate with the withheld data
# data_withheld = su.unpickle("ratings_tuple_std_withheld")

# # choose a number of features, limit the time the simulation runs
# K = 5
# max_steps = 200 # change this to something reasonable, like 200 or 500

# # update this for each trial you do with a particular k
# run = 0
# step = 180
# data_mfact = su.unpickle("mfact_%d_run_%d_%d" % (K, run, step))

# rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)
# print "================== RMSE:", 
# print rmse

# # make some predictions
# predictions = books.make_predictions(data_train, data_mfact)

# # write the predictions