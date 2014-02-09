import numpy as np
#import scipy.misc.pilutil as smp
from PIL import Image
import visualize
import kmeans

def view_rep_images(data, us, dists, n):
    min_images = []
    for x in range(len(us)):
        min_indices = [a[1] for a in mins(dists[:,x], n)]
        min_images.append(data[min_indices])
        visualize.im_show_grid(data[min_indices])
    return min_images

def mins(list, n):
    """given a list, returns the n smallest elements as a list of tuples (x, i)
    where x is the value and is the index"""
    #http://stackoverflow.com/questions/350519/getting-the-lesser-n-elements-of-a-list-in-python
    mins = [(x, i) for (i, x) in enumerate(list[:n])]
    mins.sort()
    for (i, x) in enumerate(list[n:]):
        if x < mins[-1][0]:
            mins.append((x, i + n))
            mins.sort()
            mins = mins[:n]
    return mins

def save_image(raw_data, file):
    """Takes a list of 3072 values as described on the CIFAR page
    and saves the corresponding image """
    #http://stackoverflow.com/questions/434583/what-is-the-fastest-way-to-draw-an-image-from-discrete-pixel-values-in-python
    #http://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
    data = np.array(raw_data)
    shape = (3, 1024)
    rgbData = data.reshape(shape)
    rgbData *= 255

    rgbArray = np.zeros((32,32,3), 'uint8')
    shape = (32, 32)
    rgbArray[..., 0] = rgbData[0].reshape(shape)
    rgbArray[..., 1] = rgbData[1].reshape(shape)
    rgbArray[..., 2] = rgbData[2].reshape(shape)

    #rgb_tuples = np.array(zip(*rgb_data))
    #shape = (32, 32)
    #rgb_array = rgb_tuples.reshape(shape)

    img = Image.fromarray(rgbArray)
    img.save("Sam/output/" + file + ".jpg")

    #img = smp.toimage(rgb_array)
    #img.show()