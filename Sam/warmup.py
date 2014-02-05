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

def init_Kmeans(data, k):
    return [random(0,k-1) for x in range(len(data))]

def add_lists(a,b):
    return [x + y for x, y in zip(a, b)]

def div_list_by_num(list,a):
    return [x/a for x in list]

def cluster_mean(data, k, ks):
    #num data points for each cluster
    cluster_nums = [0]*k
    #total sum of vectors for each cluster
    cluster_sums = [[0]*len(data)]*k
    for x in range(len(ks)):
        cluster_nums[ks[x]] += 1
        cluster_sums[ks[x]] = add_lists[cluster_sums[ks[x]], data[x]]
    return [div_list_by_num(cluster_sums[i],cluster_nums[i]) for i in range(k)]

def distances(data, us):
    diff = x[np.newaxis,:,:] - u[:,np.newaxis,:]
    dist = np.sum(diff**2,axis=-1)
    return dist