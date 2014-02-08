import numpy as np
import draw_image as draw
import kmeans
import load_data as load

data = load.load_CIFAR('data\warmup\cifar-10-batches-py\data_batch_1')

k=20
ks = kmeans.init_kmeans(len(data), k)


while True:
    newks = kmeans.update_cluster(data, k, ks)

    if newks == ks:
        ks = newks
        break
    else:
        ks = newks

us = kmeans.cluster_means(data, k, ks)

for (i,x) in enumerate(us):
    draw.draw_image(x, str(i))

print kmeans.objective(data, ks, us)