import numpy as np
import cPickle

def unpickle_CIFAR(file):
    """Import CIFAR datasets"""
    #http://www.cs.toronto.edu/~kriz/cifar.html
    #data: 10000x3072 numpy array
    #labels: list of 10000 numbers in the range 0-9
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_CIFAR(file):
    rawData = unpickle_CIFAR(file)
    data = np.array(rawData['data'])
    normData = data/255
    return normData.tolist()