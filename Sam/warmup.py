import numpy as np
import post_process as post
import kmeans
import load_data as load
import visualize
import copy
import shared_utils as utils

data = load.load_CIFAR('data\warmup\cifar-10-batches-py\data_batch_1')

k=10
ks = kmeans.init_kmeans(len(data), k)

objectives = []

while True:
    us = kmeans.cluster_means(data, k, ks)
    objectives.append(kmeans.objective(data, ks, us))
    newks = copy.copy(ks)
    newks = kmeans.update_cluster(data, k, newks, us)

    if newks == ks:
        ks = newks
        break
    else:
        ks = newks

us = kmeans.cluster_means(data, k, ks)
objectives.append(kmeans.objective(data, ks, us))

utils.pickle({'ks': ks, 'us': us, 'objectives': objectives}, "Sam/output/k10")
"""for (i,x) in enumerate(us):
    post.save_image(x, str(i))"""
visualize.im_show_grid(us)

#print objectives