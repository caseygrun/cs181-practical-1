import numpy as np
from scipy import *
import util
import shared_utils

ratings_filename = 'ratings_std'
mother = shared_utils.unpickle(ratings_filename)

csr_ratings = mother['ratings'].tocsr()

N = csr_ratings.shape[0]
print N

D = csr_ratings.shape[1]
print D

#dot(csr_ratings[i,:].toarray(),csr_ratings[j,:].toarray())


#csr_ratings[i,:].dot(csr_ratings[j,:])