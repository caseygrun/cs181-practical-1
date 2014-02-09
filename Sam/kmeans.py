import numpy as np
import random

def init_kmeans(len, k):
    """returns a list of cluster assignments"""
    return [random.randint(0,k-1) for x in range(len)]

def cluster_means(data, k, ks):
    """finds the mean of each cluster
    returns a k x len(data[0]) numpy array"""
    #num data points for each cluster
    kNums = np.zeros(k)
    #total sum of vectors for each cluster
    kSums = np.zeros((k, len(data[0])))

    for x in range(len(ks)):
        kNums[ks[x]] += 1
        kSums[ks[x]] += data[x]
    return kSums/kNums[:,np.newaxis]

def distances(data, us):
    """computes distance from each data point to every cluster mean"""
    #return [[list_dist(x, y) for y in us] for x in data]
    diff = data[np.newaxis,:,:] - us[:,np.newaxis,:]
    dist = np.sum(diff**2,axis=-1)
    return np.transpose(dist)

def update_cluster(data, k, ks, us):
    dist = distances(data, us)
    for i in range(len(ks)):
        val, idx = min((val, idx) for (idx, val) in enumerate(dist[i]))
        ks[i] = idx
    return ks

def objective(data, ks, us):
    sum = 0
    for i in range(len(data)):
        sum += np.sum((data[i] - us[ks[i]])**2)**2
    return sum