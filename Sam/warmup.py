import numpy as np
import draw_image as draw
import kmeans
import load_data as load
import visualize
import copy
import shared_utils

data = load.load_CIFAR('data\warmup\cifar-10-batches-py\data_batch_1')

k=10
ks = kmeans.init_kmeans(len(data), k)

while True:
    us = kmeans.cluster_means(data, k, ks)
    newks = copy.copy(ks)
    newks = kmeans.update_cluster(data, k, newks, us)

    if newks == ks:
        ks = newks
        break
    else:
        ks = newks

pickle(ks, "output/ks")

us = kmeans.cluster_means(data, k, ks)
"""for (i,x) in enumerate(us):
    draw.draw_image(x, str(i))"""
visualize.im_show_grid(us)

#print objectives