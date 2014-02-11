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
import random
import numpy as np
import scipy.sparse as sp
import shared_utils as su
import util

DEBUG = True
def debug(fmt, arg=tuple()):
	if DEBUG: print fmt % arg

def make_predictions(ratings_data, mfact_data):
	"""
	Makes a set of predictions, suitable for passing to util.write_predictions

	Arguments:
		ratings_data	: the data object returned by build_ratings
		mfact_data 		: the data object returned by mfact

	Returns:
		predictions 	: a list of dicts, suitable for passing to util.write_predictions

	"""

	# load data from the original training set
	center = ratings_data["center"]
	scale = ratings_data["scale"]
	book_isbn_to_index = ratings_data["book_isbn_to_index"]

	# load data calculated by the matrix factorization
	P = mfact_data["P"]
	Q = mfact_data["Q"]
	Bn = mfact_data["Bn"]
	Bd = mfact_data["Bd"]
	mean = mfact_data["mean"]

	# load the set of requested predictions
	queries = util.load_test("../data/books/ratings-test.csv")
	L = len(queries)
	debug("Making %d predictions",L)

	# for each query
	for (i,query) in enumerate(queries):

		# print progress
		# if DEBUG: print ("%d / %d : " % (i+1,L)),
		
		# lookup user and book index
		user_index = query["user"] - 1
		book_index = book_isbn_to_index[query["isbn"]]

		# calculate predicted rating
		rating_float = (np.dot(P[user_index,:],Q[book_index,:]) + mean + Bn[user_index] + Bd[book_index]) \
			* scale + center
		
		# coerce to range (1,5); round, convert to int
		rating = int(round(max(1,min(5,rating_float))))

		# store both values so we can do visualization of distributions later
		query["rating"] = rating
		query["rating_f"] = rating_float
		
		# print value
		# if DEBUG: print "%f -> %d" % (rating_float, rating)

	return queries

def build_ratings(filename="ratings_tuple_std", standardize=True, format="tuple", withhold=20000):
	"""
	Loads the training data, standardizes it, withholds points, and converts it
	to the desired format. This produces two dicts: `data_train` and 
	`data_withhold`. `data_train` is pickled and saved to `filename`, while 
	`data_withhold` is pickled and saved to `filename` + "_withheld"

	Arguments:

		filename	: File name to save the data to
		standardize : one of:
			-	True	: standardize the data by subtracting the mean and 
			dividing by the standard deviation
			-	'linear': standardize by subtracting 1 and dividing by 4
			-	False	: do not standardize
		withhold : number of points to withhold (must be less than the number
				of points in the training data)
		format   	: one of:
			-	"lil"	: encode the ratings as a scipy.sparse.lil_matrix
			-	"tuple"	: encode the ratings as a list of tuples (i, j, r)
			where i is the user index, j is the book index, and r is the rating

	Returns: two dicts: data_train and data_withhold; each has the following 
	keys. 

		-	"ratings" : a list of tuples (i,j,r) where i is the user, j is the 
			book, and r is the rating that the user gave the book
		-	"book_isbn_to_index" : dict that maps the ISBN for each book to a 
			numerical index j.
		-	"N" : the number of users
		-	"D" : the number of books
		-	"T" : the number of ratings
		-	"center" : the number used to offset the ratings (for standardization)
		-	"scale" : the number used to scale the ratings (for standardization);
			The ratings can be de-standardized by: 
			destandardized = standardized * scale + center
		-	"mean" : the mean of the ratings
		-	"variance" : the variance of the ratings
	"""

	# load ratings from CSV into list of tuples
	data = build_ratings_tuple()

	# partition ratings
	(data_train, data_withhold) = partition_ratings(data,withhold)

	# standardize ratings
	print "Training data: "
	data_train = standardize_ratings(data_train, standardize)
	
	print "Withheld data: "
	data_withhold = standardize_ratings(data_withhold, False)

	# convert to LIL if desired
	if(format=="lil"):
		data_train["ratings"] = ratings_tuple_to_lil(data_train["ratings"], data_train["N"], data_train["D"])
		data_withhold["ratings"] = ratings_tuple_to_lil(data_withhold["ratings"], data_withhold["N"], data_withhold["D"])

	# save
	print "Saving training data to %s" % filename
	su.pickle(data_train,filename)

	filename_withheld = filename+"_withheld"
	print "Saving withheld data to %s" % filename_withheld
	su.pickle(data_withhold,filename_withheld)

def build_ratings_tuple():
	"""
	Loads the training data for N users and D books, and builds a list of tuples 

	Returns: a dict with the following fields:

		-	`ratings` : a list of tuples (i,j,r) where i is the user, j is the 
			book, and r is the rating that the user gave the book
		-	`book_isbn_to_index` : dict that maps the ISBN for each book to a 
			numerical index j.
		-	`N` : the number of users
		-	`D` : the number of books
		-	`T` : the number of ratings in the training set
	"""

	print "Loading Users..."
	users = util.load_users("../data/books/users.csv")
	user_ids = sorted([ user["user"] for user in users ])
	N = len(user_ids)
	del users
	print "Loaded %d users." % N


	print "Loading Books..."
	books = util.load_books("../data/books/books.csv")
	book_isbns = sorted([ book["isbn"] for book in books ])
	book_isbn_to_index = dict( zip(book_isbns,range(len(book_isbns))) )
	D = len(book_isbns)
	print "Loaded %d books." % D


	print "Loading Trainings..."
	train = util.load_train("../data/books/ratings-train.csv")
	T = len(train)
	print "Loaded %d ratings." % T
	ratings = [(rating["user"]-1, book_isbn_to_index[rating["isbn"]], (rating["rating"])) for rating in train]

	return { "ratings": ratings, "book_isbn_to_index": book_isbn_to_index , \
		"N": N, "D": D, "T": T}

def partition_ratings(data,withhold):
	"""
	Partitions a list of rankings randomly into a set to be trained and a set
	to be withheld.

	Arguments:
		data 		: data object of the form returned by build_ratings_tuple
		withhold	: number of ratings to withhold (must be < len(ratings))

	Returns:
		data_train	: data object to train on
		data_withhold : data object to withhold

	"""

	# create clone of data for withholding
	data_train = data
	data_withhold = data.copy()
	ratings = data["ratings"]

	print "Witholding %d ratings, training on %d" % (withhold, len(ratings)-withhold)

	# randomly permute ratings
	random.shuffle(ratings)

	# partition ratings into training set and withheld set
	data_withhold["ratings"] = ratings[0:withhold]
	data_withhold["T"] = len(data_withhold["ratings"])

	data_train["ratings"] = ratings[withhold:]
	data_train["T"] = len(data_train["ratings"])

	return (data_train, data_withhold)

def standardize_ratings(data, mode=True):
	"""
	Standardizes a set of ratings 

	Arguments:

		data	: dict of the form returned by build_ratings_tuple
		mode	: one of:
			-	True	: standardize the data by subtracting the mean and 
			dividing by the standard deviation
			-	'linear': standardize by subtracting 1 and dividing by 4
			-	False	: do not standardize

	Returns:

		data	: dict of the form returned by build_ratings, but augmented 
			with the following keys
			-	"center"	: value subtracted from each rating 
			-	"scale"		: rating is divided by the sqrt of this value
			-	"mean"		: mean of the training data
			-	"variance"	: variance of the training data

	"""

	print "Calculating mean and variance..."
	T = len(data["ratings"])
	x = 0.0
	x2 = 0.0
		
	for (i,j,Rij) in data["ratings"]:
		x += Rij
		x2 += Rij**2

	mean = x / T
	var = (x2/T) - mean**2
	std = math.sqrt(var)
	print "Mean: %f , Variance: %f , Std. Dev: %f" % (mean, var, std)

	if(mode == True):

		# build up the sum of the values and the squared values of each rating, in order to calculate the mean and variance
		print "Standardizing ratings..."
		center = mean
		scale = std

	elif (mode == 'linear'):
		
		print "Standardizing ratings to the interval [0,1]..."
		center = 1
		scale = 4

	else:

		print "No standardization applied to ratings."
		center = 0
		scale = 4

	data["ratings"] = [ (i, j, (Rij - center)/scale) for (i, j, Rij) in data["ratings"]]
	data["center"] = center
	data["scale"] = scale
	data["mean"] = mean
	data["variance"] = var
	return data

def ratings_tuple_to_lil(ratings_tuple, N, D):
	"""
	Converts a list of tuples to an N x D 
	scipy.sparse.lil_matrix
	"""
	ratings = sp.lil_matrix((N,D))
	for (i, j, rating) in ratings_tuple:
		ratings[i,j] = rating
	return ratings
