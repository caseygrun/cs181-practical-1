import numpy as np
import random

def init_kmeans(len, k):
    """returns a list of cluster assignments"""
    return [random.randint(0,k-1) for x in range(len)]

def add_lists(a,b):
    return [x + y for x, y in zip(a, b)]

def div_list_by_num(list,a):
    if a == 0:
        return list
    else:
        return [x/a for x in list]

def cluster_means(data, k, ks):
    """finds the mean of each cluster
    returns a k x len(data[0]) numpy array"""
    #num data points for each cluster
    kNums = np.zeros(k)
    #total sum of vectors for each cluster
    kSums = np.zeros((k, len(data[0])))

    for x in range(len(ks)):
        kNums[ks[x]] += 1
        #ksSums[ks[x]] = add_lists(kSums[ks[x]], data[x])
        kSums[ks[x]] += data[x]
    #return [div_list_by_num(ksSums[i], kNums[i]) for i in range(k)]
    return kSums/kNums[:,np.newaxis]

def distances(data, us):
    """computes distance from each data point to every cluster mean"""
    #return [[list_dist(x, y) for y in us] for x in data]
    diff = data[np.newaxis,:,:] - us[:,np.newaxis,:]
    dist = np.sum(diff**2,axis=-1)
    return np.transpose(dist)

def list_dist(a, b):
    diff = [x - y for x, y in zip(a, b)]
    return sum([x**2 for x in diff])

def update_cluster(data, k, ks):
    dist = distances(data, cluster_means(data, k, ks))
    for i in range(len(ks)):
        val, idx = min((val, idx) for (idx, val) in enumerate(dist[i]))
        ks[i] = idx
    return ks