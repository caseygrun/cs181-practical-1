import numpy as np

def unpickle_CIFAR(file):
    """Import CIFAR datasets"""
    #http://www.cs.toronto.edu/~kriz/cifar.html
    #data: 10000x3072 numpy array
    #labels: list of 10000 numbers in the range 0-9
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def init_kmeans(data, k):
    return [random(0,k-1) for x in range(len(data))]

def add_lists(a,b):
    return [x + y for x, y in zip(a, b)]

def div_list_by_num(list,a):
    return [x/a for x in list]

def cluster_mean(data, ks):
    k=len(ks)
    #num data points for each cluster
    cluster_nums = [0]*k
    #total sum of vectors for each cluster
    cluster_sums = [[0]*len(data)]*k
    for x in range(len(ks)):
        cluster_nums[ks[x]] += 1
        cluster_sums[ks[x]] = add_lists[cluster_sums[ks[x]], data[x]]
    return [div_list_by_num(cluster_sums[i],cluster_nums[i]) for i in range(k)]

def distances(data, us):
    diff = data[np.newaxis,:,:] - us[:,np.newaxis,:]
    dist = np.sum(diff**2,axis=-1)
    return dist

def update_cluster(data, ks):
    dist = distances(data, cluster_mean(data, ks))
    for i in range(len(ks)):
        val, idx = min((val, idx) for (idx, val) in enumerate(dist(i)))
        ks[i] = idx
    return ks

raw_data = unpickle_CIFAR('data_batch_1')
data = raw_data['data']
k=10
ks = init_kmeans(data, k)

while True:
    us = cluster_mean(data, ks)
    newks = update_cluster(data, ks)

    if newks == ks:
        break
    else:
        ks = newks