import numpy as np
import util

pred_filename  = 'pred-user-kmeans.csv'
train_filename = 'ratings-train.csv'
test_filename  = 'ratings-test.csv'

training_data  = util.load_train(train_filename)
test_queries   = util.load_test(test_filename)

for query in test_queries:
	book = query['isbn']

util.write_predictions(test_queries, pred_filename)