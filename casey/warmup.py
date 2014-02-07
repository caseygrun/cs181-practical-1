# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Warmup
# Casey Grun
# 
# warmup.py
# Calls the k-means clustering algorithm defined in kmeans.py
# ----------------------------------------------------------------------------

import numpy as np
import kmeans

# load one part of the CIFAR-10 dataset
d = kmeans.unpickle("../data/warmup/cifar-10-batches-py/data_batch_1")
X = kmeans.standardize(d["data"])

(U,R) = kmeans.cluster(X,10,kmeans.dist)
print U
print R

kmeans.pickle(U,"U")
kmeans.pickle(R,"R")

np.savetxt("U.csv", U)
np.savetxt("R.csv", R)




# # Try a simple example that's easily verified
# X = np.array([[10, 0, 0, 0],
# 			  [11, 0, 0, 0],
# 			  [9 , 0, 0, 0],
# 			  [0 , 8, 0, 0],
# 			  [0 , 6, 0, 0],
# 			  [0 , 7, 0, 0],
# 			  [0 , 0, 9, 0],
# 			  [0 , 0, 8, 0],
# 			  [0 , 0,10, 0],])
# X = np.array([[10, 0, 0, 0],
# 			  [11, 0, 0, 0],
# 			  [9 , 0, 0, 0],
# 			  [0 , 8, 0, 0],
# 			  [0 , 6, 0, 0],
# 			  [0 , 7, 0, 0],
# 			  [0 , 0, 9, 0],
# 			  [0 , 0, 8, 0],
#			  [0 , 0,10, 0],])
# (U, R) = kmeans.cluster(X,3,dist)

# print U
# print R

