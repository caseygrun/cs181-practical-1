import numpy as np
import util
import shared_utils

pred_filename  = 'pred-user-mean.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)



for query in test_queries:
	user = query['user']
	user_cluster = numpy.dot(R[user - 1,:], range(k))
	isbn = query['isbn']
	book_index = mother['book_isbn_to_index'][isbn]
	query['rating'] = U[user_cluster][book_index] * mother['variance'] + mother['mean']

util.write_predictions(test_queries, pred_filename)