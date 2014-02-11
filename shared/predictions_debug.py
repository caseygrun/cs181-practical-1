import sys
import math
import numpy as np
import books
import visualize
import mf_debug
import util
import shared_utils as su


# do this once to build the ratings and save them to ratings_tuple_std
books.build_ratings(filename="ratings_tuple_std", standardize=True, withhold=0)

# load training data
data_train = su.unpickle("ratings_tuple_std")

# choose a number of features, limit the time the simulation runs
K = 1
max_steps = 1000 # change this to something reasonable, like 200 or 500

# update this for each trial you do with a particular k
run = 0

data_mfact = mf_debug.mfact(data_train["ratings"], data_train["N"], data_train["D"], \
    K, steps=max_steps, filename=("debug_data/mfact_%d_run_%d" % (K, run)))

# make some predictions
predictions = books.make_predictions(data_train, data_mfact)

# write the predictions
util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))