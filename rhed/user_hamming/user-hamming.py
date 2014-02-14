import numpy
import util
import shared_utils

pred_filename  = 'pred-user-hamming.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)

user_common_books = shared_utils.unpickle('user_common_books')
user_difference_ratings = shared_utils.unpickle('user_difference_ratings')

print user_common_books[0:]
print user_difference_ratings[0:]

ratings_filename = 'ratings_std'
mother = shared_utils.unpickle(ratings_filename)
'''
for query in test_queries:
	user = query['user']
	user_cluster = numpy.dot(R[user - 1,:], range(k))
	isbn = query['isbn']
	book_index = mother['book_isbn_to_index'][isbn]
	query['rating'] = U[user_cluster][book_index] * mother['variance'] + mother['mean']

util.write_predictions(test_queries, pred_filename)
'''