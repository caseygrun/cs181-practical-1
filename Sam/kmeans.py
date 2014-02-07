import numpy as np
import random

def init_kmeans(data, k):
    return [random.randint(0,k-1) for x in range(len(data))]

def add_lists(a,b):
    return [x + y for x, y in zip(a, b)]

def div_list_by_num(list,a):
    if a == 0:
        return list
    else:
        return [x/a for x in list]

def cluster_mean(data, k, ks):
    #num data points for each cluster
    cluster_nums = [0]*k
    #total sum of vectors for each cluster
    cluster_sums = [[0]*len(data)]*k
    for x in range(len(ks)):
        cluster_nums[ks[x]] += 1
        cluster_sums[ks[x]] = add_lists(cluster_sums[ks[x]], data[x])
    return [div_list_by_num(cluster_sums[i], cluster_nums[i]) for i in range(k)]

def distances(data, us):
    return [[list_dist(x, y) for y in us] for x in data]

def list_dist(a, b):
    diff = [x - y for x, y in zip(a, b)]
    return sum([x**2 for x in diff])

def update_cluster(data, k, ks):
    dist = distances(data, cluster_mean(data, k, ks))
    for i in range(len(ks)):
        val, idx = min((val, idx) for (idx, val) in enumerate(dist[i]))
        ks[i] = idx
    return ks