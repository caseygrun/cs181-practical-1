import numpy as np
#import scipy.misc.pilutil as smp
from PIL import Image

def draw_image(raw_data):
    """Takes a list of 3072 values as described on the CIFAR page
    and outputs the corresponding image """
    #http://stackoverflow.com/questions/434583/what-is-the-fastest-way-to-draw-an-image-from-discrete-pixel-values-in-python
    #http://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
    data = np.array(raw_data)
    print data.shape
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
    img.save('test.jpg')

    #img = smp.toimage(rgb_array)
    #img.show()