import sys
import math
import numpy as np
import books
import visualize
import mf_debug as mf
import util
import shared_utils as su


# do this once to build the ratings and save them to ratings_tuple_std
#books.build_ratings(filename="ratings_tuple_std", standardize=True, withhold=80000)

# load training data
data_train = su.unpickle("ratings_tuple_std")
data_withheld = su.unpickle("ratings_tuple_std_withheld")

rmses = []

K = 1
max_steps = 200
run = 2
data_mfact = mf.mfact_cont(data_train["ratings"], data_train["N"], data_train["D"], \
        K, 'debug_data/mfact_1_run_1_180', steps=max_steps, filename=("debug_data/mfact_%d_run_%d" % (K, run)))

rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)
print rmse

# make some predictions
predictions = books.make_predictions(data_train, data_mfact)

# write the predictions
util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))

"""for x in [1, 2, 3, 4, 5]:

    # choose a number of features, limit the time the simulation runs
    K = x
    max_steps = 200 # change this to something reasonable, like 200 or 500

    # update this for each trial you do with a particular k
    run = x

    data_mfact = mf.mfact(data_train["ratings"], data_train["N"], data_train["D"], \
        K, steps=max_steps, filename=("debug_data/mfact_%d_run_%d" % (K, run)))

    # make some predictions
    #predictions = books.make_predictions(data_train, data_mfact)

    # write the predictions
    #util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))


    #data_mfact = su.unpickle("debug_data/mfact_1_run_0_200")

    rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)
    print rmse
    rmses.append(rmse)
    print rmses"""


"""data_mfact = su.unpickle("debug_data/mfact_1_run_1_180")

P = data_mfact['P']
means = books.user_mean(data_train)

for userI in range(len(P)):
    print P[userI], means[userI]"""