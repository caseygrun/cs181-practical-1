# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# predictions2.py
# Does book predictions based on the data calculated in predictions.py
# 
# ----------------------------------------------------------------------------

import sys
import math
import numpy as np
import books
import kmeans
import pca
import visualize
import util
import shared_utils as su

# load training data
d = su.unpickle("output/ratings_tuples")
ratings = d["ratings"]
N = d["N"]
D = d["D"]
book_isbn_to_index = d["book_isbn_to_index"]
# this is the mean of the un-standardized training data, used to un-standardize
# the data at the end
mean = d["mean"]
var = d["var"]
std = math.sqrt(var)

# do prediction based on matrix factorization
K = 20
run = 2
step = 260
mfact = su.unpickle("output/mfact_%d_run_%d/mfact_%d_%d" % (K, run, K, step))
P = mfact["P"]
Q = mfact["Q"]
Bn = mfact["Bn"]
Bd = mfact["Bd"]
# this is the mean of the standardized training data, used for the learning/
# prediction
standard_mean = mfact["mean"] 

# load the set of requested predictions
queries = util.load_test("../data/books/ratings-test.csv")
L = len(queries)

# for each query
for (i,query) in enumerate(queries):
	print ("%d / %d : " % (i+1,L)),
	user_index = query["user"] - 1
	book_index = book_isbn_to_index[query["isbn"]]

	# calculate predicted rating
	rating_float = (np.dot(P[user_index,:],Q[book_index,:]) + standard_mean + Bn[user_index] + Bd[book_index]) * std + mean
	
	# coerce to range (1,5); round, convert to int
	rating = int(round(max(1,min(5,rating_float))))

	# store both values so we can do visualization of distributions later
	query["rating"] = rating
	query["rating_f"] = rating_float
	print "%f -> %d" % (rating_float, rating)

su.pickle(queries,"output/mfact_%d_run_%d/predictions" % (K,run))
util.write_predictions(queries, "output/mfact_%d_run_%d/predictions.csv" % (K,run))

# ----------------------------------------------------------------------------

# # visualize distribution of ratings
# queries = su.unpickle("output/mfact_%d_run_%d/predictions" % (K,run))

# ratings = []
# ratings_f = []
# for query in queries:
# 	ratings.append(query["rating"])
# 	ratings_f.append(float(query["rating_f"]))

# # print ratings_f
# # sys.exit(0)

# print np.mean(ratings), np.var(ratings)
# print np.mean(ratings_f), np.var(ratings_f)

# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.hist(ratings,5)
# plt.subplot(1,2,2)
# plt.hist(ratings_f,20)
# plt.show()