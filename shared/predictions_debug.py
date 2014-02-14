import sys
import math
import numpy as np
import books
import visualize
import mf_debug as mf
import util
import shared_utils as su


# do this once to build the ratings and save them to ratings_tuple_std
books.build_ratings(filename="ratings_tuple_std", standardize=False, withhold=20000)

# load training data
data_train = su.unpickle("ratings_tuple_std")
data_withheld = su.unpickle("ratings_tuple_std_withheld")

"""rmses=[]
#for K in [2,3,4,5]:
K = 4
max_steps = 1000
run = 1
data_mfact = mf.mfact(data_train["ratings"], data_train["N"], data_train["D"], \
        K, steps=max_steps, save_every=50, filename=("debug_data/mfact_%d_run_%d" % (K, run)))
#data_mfact = mf.mfact_cont(data_train["ratings"], data_train["N"], data_train["D"], \
#        K, 'debug_data/mfact_1_run_1_80', steps=max_steps, filename=("debug_data/mfact_%d_run_%d" % (K, run)))
#data_mfact = su.unpickle("debug_data/mfact_3_run_1_650")
rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)
print rmse
rmses.append(rmse)
print rmses"""

# make some predictions
#predictions = books.make_predictions(data_train, data_mfact)

# write the predictions
#util.write_predictions(predictions,("predictions_%d_run_%d.csv" % (K, run)))


""""P = data_mfact['P']
means = books.user_mean(data_train)

for userI in range(len(P)):
    print P[userI], means[userI]"""
K=1
max_steps = 500
run = 2
data_mfact = mf.mfact_kfixed(data_train["ratings"], data_train["N"], data_train["D"], \
        K, steps=max_steps, filename=("debug_data/mfact_%d_run_%d" % (K, run)))
#data_mfact = su.unpickle("debug_data/mfact_1_run_1_40")
#data_mfact = mf.mfact_kfixed(data_train["ratings"], data_train["N"], data_train["D"], \
#        K, steps=max_steps, save_every=50, filename=("debug_data/mfact_%d_run_%d" % (K, run)), contFile='debug_data/mfact_1_run_1_50',)
#data_mfact = mf.mfact_pfixed(data_train["ratings"], data_train["N"], data_train["D"], \
#        K, steps=max_steps, save_every=25, filename=("debug_data/mfact_%d_run_%d" % (K, run)), contFile='debug_data/mfact_1_run_1_50',)

rmse = books.rmse_withheld(data_train, data_withheld, data_mfact)
print rmse