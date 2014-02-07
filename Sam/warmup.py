import numpy as np
import draw_image as draw
import kmeans
import load_data as load

#TODO: clean code, separate into different files
#fix div/0 error


data = load.load_CIFAR('C:\Users\Sam\git\cs181-practical-1\Sam\data_batch_1')

"""data = np.array([[10, 0, 0, 0],
              [11, 0, 0, 0],
              [9 , 0, 0, 0],
              [0 , 8, 0, 0],
              [0 , 6, 0, 0],
              [0 , 7, 0, 0],
              [0 , 0, 9, 0],
              [0 , 0, 8, 0],
              [0 , 0,10, 0],])"""
k=10
ks = kmeans.init_kmeans(len(data), k)


while True:
    newks = kmeans.update_cluster(data, k, ks)

    if newks == ks:
        break
    else:
        ks = newks

#print ks
for (i,x) in enumerate(kmeans.cluster_means(data, k, ks)):
    draw.draw_image(x, str(i))