# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# predictions.py
# Does book predictions and calculates data using k-means, PCA, or SVD
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

# # build ratings
# books.build_ratings(filename="output/ratings_std",standardize=True)
# books.build_ratings(filename="output/ratings_tuples",standardize=False,format="tuples")
# books.build_ratings(filename="output/ratings_tuples_linear",standardize='linear',format="tuples")
# sys.exit(0)
# (ratings, book_isbn_to_index) = books.load_ratings()
# ratings = ratings.tocsr()

# visualize.show_sparsity(ratings)
# sys.exit(0)

# --------------------------------------------------------------------

# do K-means clustering
# K = 20
# (U,R) = kmeans.cluster2(ratings, K, kmeans.dist2)
# print U
# print R

# kmeans.pickle(U,"output/U_%d_std" % K)
# kmeans.pickle(R,"output/R_%d_std" % K)

# np.savetxt("output/U_%d_std.csv" % K, U)
# np.savetxt("output/R_%d_std.csv" % K, R)

# --------------------------------------------------------------------


# # do PCA
# (w,v) = pca.analyze(ratings,2)

# kmeans.pickle((w,v),"output/WV")
# np.savetext("output/W.csv",w)
# np.savetext("output/V.csv",v)

# --------------------------------------------------------------------

# # do SVD
# K = 2
# (u, s, vt) = pca.svd(ratings, K)
# print u
# print s
# print vt

# kmeans.pickle((u,s,vt),"output/svd_%d" % K)
# (u, s, vt) = kmeans.unpickle("output/svd_%d" % K)

# print "Projecting ratings on to %d dimensions" % K
# ratings_p = pca.project2(ratings, vt)

# import matplotlib.pyplot as plt

# # scatterplot
# plt.scatter(ratings_p[:,0],ratings_p[:,1])
# plt.show()

# # histogram
# plt.subplot(1,2,1)
# plt.hist(ratings_p[:,0],50)
# plt.subplot(1,2,2)
# plt.hist(ratings_p[:,1],50)
# plt.show()

# # graph to find where in the list the outliers are 
# plt.subplot(1,2,1)
# plt.plot(ratings_p[:,0])
# plt.subplot(1,2,2)
# plt.plot(ratings_p[:,1])
# plt.show()

# --------------------------------------------------------------------

# do matrix factorization
# (ratings, book_isbn_to_index) = books.load_ratings("output/ratings_tuples")
# ratings = ratings.tocsr()
# print ratings
# sys.exit(0)

# load training data
d = su.unpickle("output/ratings_tuples_linear")
ratings = d["ratings"]
N = d["N"]
D = d["D"]
book_isbn_to_index = d["book_isbn_to_index"]
mean = d["mean"]
std = math.sqrt(d["var"])

# do matrix factorization
K = 30
mfact = pca.mfact2(ratings,N,D,K)
# su.pickle(mfact,"output/mfact_%d" % K)
sys.exit(0)

# copy all the generated mfact_%d files into a subdirectory output/mfact_K_run_R,
# replacing K with the value for K above, and R with the number of times you've
# run the script for that K. 
# then use predictions2.py to calculate the real predictions
