#----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
#
# predictions.py
# Does book predictions and calculates data using k-means, PCA, or SVD
#
# ----------------------------------------------------------------------------

import sys
import numpy as np
import books
import kmeans
import koren
import visualize
import util
import shared_utils as su

# load training data
d = su.unpickle("output/ratings_tuples")
ratings = d["ratings"]
N = d["N"]
D = d["D"]
book_isbn_to_index = d["book_isbn_to_index"]

# # do matrix factorization
K = 20
mfact = koren.mfact2(ratings,N,D,K)
su.pickle(mfact,"output/mfact_%d" % K)
sys.exit(0)

"""
# do prediction based on matrix factorization
K = 20
run = 0
step = 20
mfact = su.unpickle("output/mfact_%d_run_%d/mfact_%d_%d" % (K, run, K, step))
P = mfact["P"]
Q = mfact["Q"]
Bn = mfact["Bn"]
Bd = mfact["Bd"]
mean = mfact["mean"]

queries = util.load_test("../data/books/ratings-test.csv")
L = len(queries)
for (i,query) in enumerate(queries):
    print ("%d / %d" % (i,L)),
    user_index = query["user"] - 1
    book_index = book_isbn_to_index[query["isbn"]]
    # calculate rating
    rating = (np.dot(P[user_index,:],Q[book_index,:]) + mean + Bn[user_index] + Bd[book_index])
    # coerce to range (1,5)
    rating = max(1,min(5,rating))
    # convert to int
    rating = int(round(rating))
    query["rating"] = rating
    print query

util.write_predictions(queries, "output/mfact_%d_run_%d/predictions.csv" % (K,run))
thon/"""