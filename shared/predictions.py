# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# predictions.py
# Shows an example of how to do book predictions using matrix factorization
# ----------------------------------------------------------------------------

import sys
import math
import numpy as np
import books
import visualize
import mf
import util
import shared_utils as su


# do this once to build the ratings and save them to ratings_tuple_std
books.build_ratings(filename="ratings_tuple_std", standardize=True, withhold=20000)

# load training data
data_train = su.unpickle("ratings_tuple_std")

# choose a number of features, limit the time the simulation runs
K = 5
max_steps = 2 # change this to something reasonable, like 200 or 500

# update this for each trial you do with a particular k
run = 0

data_mfact = mf.mfact(data_train["ratings"], data_train["N"], data_train["D"], \
	K, steps=max_steps, filename=("mfact_%d_run_%d" % (K, run)))

# make some predictions
predictions = books.make_predictions(data_train, data_mfact)

# write the predictions
util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))