import numpy as np
import util
import shared_utils

pred_filename  = 'pred-user-kmeans.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)

k = 20

u_filename = 'U_' + str(k) + '_std'
r_filename = 'R_' + str(k) + '_std'

U = shared_utils.unpickle(u_filename)
R = shared_utils.unpickle(r_filename)

ratings_filename = 'ratings_std'
mother = shared_utils.unpickle(ratings_filename)

for query in test_queries:
	user = query['user']
	user_cluster = np.dot(R[user - 1,:], range(k))
	isbn = query['isbn']
	book_index = mother['book_isbn_to_index'][isbn]
	query['rating'] = U[user_cluster][book_index] * mother['variance'] + mother['mean']

util.write_predictions(test_queries, pred_filename)