import numpy as np
import post_process as post
import load_data as load
import shared_utils as utils
import visualize
import matplotlib.pyplot as plt
import kmeans

#data = load.load_CIFAR('data\warmup\cifar-10-batches-py\data_batch_1')
#utils.pickle({'ks': ks, 'us': us, 'objectives': objectives}, "Sam/output/k10")
results = utils.unpickle('Sam/output/kplus10')
#utils.pickle(kmeans.distances(data, us),'Sam/output/k10dists')
#dists = utils.unpickle('Sam/output/k10dists')

ks = results['ks']
us = results['us']
#objectives = results['objectives']
print results['objective']

#visualize.im_show_grid(us)

"""x = np.array(range(len(objectives)))
y = np.array(objectives)
plt.plot(x,y)
plt.show()"""

#post.view_rep_images(data, us, dists,25)

