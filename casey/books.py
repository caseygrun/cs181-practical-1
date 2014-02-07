import sys
import numpy as np
import scipy.sparse as sp
import util
import kmeans

def build_ratings():
	print "Loading Users..."
	users = util.load_users("../data/books/users.csv")

	user_ids = sorted([ user["user"] for user in users ])
	# user_index_to_id = dict( zip(range(len(user_ids)),user_ids) )
	# user_id_to_index = dict( zip(user_ids,range(len(user_ids))) )
	N = len(user_ids)
	del users
	print "Loaded %d users." % N


	print "Loading Books..."
	books = util.load_books("../data/books/books.csv")

	book_isbns = sorted([ book["isbn"] for book in books ])
	# book_index_to_isbn = dict( zip(range(len(book_isbns)),book_isbns) )
	book_isbn_to_index = dict( zip(book_isbns,range(len(book_isbns))) )
	D = len(book_isbns)
	print "Loaded %d books." % D



	print "Loading Trainings..."
	train = util.load_train("../data/books/ratings-train.csv")

	print "Building ratings matrix..."

	T = len(train)
	i = 0
	ratings = sp.lil_matrix((N,D))

	for rating in train:
		ratings[ rating["user"]-1, book_isbn_to_index[rating["isbn"]] ] = rating["rating"]
		print "%d / %d" % (i, T)
		i = i+1

	kmeans.pickle({ "ratings": ratings, "book_isbn_to_index": book_isbn_to_index },"ratings")

def load_ratings():
	d = kmeans.unpickle("ratings")
	return (d["ratings"], d["book_isbn_to_index"])

(ratings, book_isbn_to_index) = load_ratings()

ratings = ratings.tocsr()

(U,R) = kmeans.cluster2(ratings, 3, kmeans.dist2)
print U
print R

kmeans.pickle(U,"U")
kmeans.pickle(R,"R")

np.savetxt("U.csv", U)
np.savetxt("R.csv", R)

# import matplotlib.pyplot as plt

# plt.spy(ratings, precision=1e-3, marker='.', markersize=5)
# plt.show()
