# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# predictions.py
# Does book predictions and calculates data using k-means, PCA, or SVD
# 
# ----------------------------------------------------------------------------


import numpy as np
import books
import kmeans
import pca
import visualize

# books.build_ratings()
(ratings, book_isbn_to_index) = books.load_ratings()
ratings = ratings.tocsr()

# visualize.show_sparsity(ratings)

# --------------------------------------------------------------------

# do K-means clustering
K = 20
(U,R) = kmeans.cluster2(ratings, K, kmeans.dist2)
print U
print R

kmeans.pickle(U,"output/U_%d_std" % K)
kmeans.pickle(R,"output/R_%d_std" % K)

np.savetxt("output/U_%d_std.csv" % K, U)
np.savetxt("output/R_%d_std.csv" % K, R)

# --------------------------------------------------------------------


# # do PCA
# (w,v) = pca.analyze(ratings,2)

# kmeans.pickle((w,v),"output/WV")
# np.savetext("output/W.csv",w)
# np.savetext("output/V.csv",v)

# --------------------------------------------------------------------

# # do SVD
# K = 10
# (u, s, vt) = pca.svd(ratings, K)
# print u
# print s
# print vt

# kmeans.pickle((u,s,vt),"output/svd_%d" % K)
