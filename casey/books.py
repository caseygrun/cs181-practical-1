# ----------------------------------------------------------------------------
# CS 181 | Practical 1 | Predictions
# Casey Grun
# 
# books.py
# Contains shared utility functions for the books problem
# 
# ----------------------------------------------------------------------------


import sys
import math
import numpy as np
import scipy.sparse as sp
import shared_utils as su
import util
import kmeans
import pca

def build_ratings(filename="ratings_std", standardize=True, format="lil"):
	"""
	Loads the training data for N users and D books, and builds an (N x D) 
	array to store the ratings. 

	Arguments:

		filename	: File name to save the data to
		standardize	: one of:
			-	True	: standardize the data by subtracting the mean and 
			dividing by the standard deviation
			-	'linear': standardize by subtracting 1 and dividing by 4
			-	False	: do not standardize
		format   	: one of:
			-	"lil"	: encode the ratings as a scipy.sparse.lil_matrix
			-	"tuple"	: encode the ratings as a list of tuples (i, j, r)
			where i is the user index, j is the book index, and r is the rating

	Returns: a dict with the following fields, and
	pickles it to the given `filename`. 

		*	`ratings` : (N x D) matrix of ratings. Ratings are standardized to
			have overall mean 0 and variance 1. 
		*	`mean` : mean of all ratings
		*	`variance` : variance of all ratings
		*	`book_isbn_to_index` : maps the ISBN for each book to a numerical
			index.

	"""
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



	# build up the sum of the values and the squared values of each rating, in order to calculate the mean and variance
	T = len(train)
	
	if(standardize == True):
		print "Standardizing ratings..."

		x = 0
		x2 = 0
		
		for rating in train:
			x += rating["rating"]
			x2 += rating["rating"]**2

		mean = x / T
		var = (x2/T) - mean**2
		std = math.sqrt(var)
		print "Mean: %d , Variance: %d , Std. Dev: %d" % (mean, var, std)
	elif (standardize == 'linear'):
		mean = 1
		std = 4.
		var = 2.
	else:
		mean = 0
		var = 1
		std = 1


	print "Building ratings matrix..."
	print "0 / %d" % T

	# a sparse matrix to hold the ratings
	if format == "lil":
		ratings = sp.lil_matrix((N,D))

		for (i, rating) in enumerate(train):
			ratings[ rating["user"]-1, book_isbn_to_index[rating["isbn"]] ] = (rating["rating"] - mean) / std
			print "%d / %d" % (i, T)
	elif format =="tuples":
		ratings = [(rating["user"]-1, book_isbn_to_index[rating["isbn"]], (rating["rating"] - mean) / std) for rating in train]

	print ratings
	su.pickle({ "ratings": ratings, "book_isbn_to_index": book_isbn_to_index , "mean": mean, "variance": var, "N": N, "D": D, "T": T},filename)

def load_ratings(filename="output/ratings_std"):
	"""
	Loads the ratings data. Gives you a tuple (ratings, book_isbn_to_index), 
	where `ratings` and `book_isbn_to_index` are described in the 
	`build_ratings` function.
	"""
	d = su.unpickle(filename)
	return (d["ratings"], d["book_isbn_to_index"])

