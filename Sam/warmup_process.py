import numpy as np
import post_process as post
import load_data as load
import shared_utils as utils
import visualize
import matplotlib.pyplot as plt
import kmeans

data = load.load_CIFAR('data\warmup\cifar-10-batches-py\data_batch_1')
#utils.pickle({'ks': ks, 'us': us, 'objectives': objectives}, "Sam/output/k10")
results = utils.unpickle('Sam/output/k10')
#utils.pickle(kmeans.distances(data, us),'Sam/output/k10dists')
dists = utils.unpickle('Sam/output/k10dists')

ks = results['ks']
us = results['us']
objectives = results['objectives']

"""print us
visualize.im_show_grid(us*255)

x = np.array(range(len(objectives)))
y = np.array(objectives)
plt.plot(x,y)
plt.show()"""

post.view_rep_images(data, us, dists,10)

