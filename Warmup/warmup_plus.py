import numpy as np
import post_process as post
import kmeans
import kmeansplus as plus
import load_data as load
import visualize
import copy
import shared_utils as utils

data = load.load_CIFAR('data_batch_1')

k=10

results = plus.cluster(data, k, utils.dist)

us = results[0]
rs = results[1]
ks = [np.where(rs[x]==1) for x in range(len(rs))]
print len(ks)
objective = kmeans.objective(data, ks, us)

utils.pickle({'ks': ks, 'us': us, 'objective': objective}, "k5")
utils.pickle(kmeans.distances(data, us),'k5dists')
"""for (i,x) in enumerate(us):
    post.save_image(x, str(i))"""
visualize.im_show_grid(us)

print objective