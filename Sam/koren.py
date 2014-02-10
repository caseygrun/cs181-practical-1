import numpy as np
import util

def stoch_grad_desc(rs, f, qs, ps, a, b):
    for (p,q,r) in rs:
        e = r - np.dot(qs[q],ps[p])
        qs[q] += a * (e * ps[p] - b * qs[q])
        ps[q] += a * (e * qs[p] - b * ps[q])

def init_ratings():
    return

def book_biases():
    train_filename = 'ratings-train.csv'
    book_filename  = 'books.csv'

    training_data  = util.load_train(train_filename)
    book_list      = util.load_books(book_filename)

    books = {}
    for book in book_list:
        books[book['isbn']] = { 'total': 0, # For storing the total of ratings.
                            'count': 0, # For storing the number of ratings.
                            }

    # Iterate over the training data to compute means.
    for rating in training_data:
        books[rating['isbn']]['total'] += rating['rating']
        books[rating['isbn']]['count'] += 1

    bBooks = np.zeros(len(book_list))

    for book in book_list:
        isbn = book['isbn']
        bBooks[isbnIndex[isbn]] = float(book['total']) / book['count']
    float(book['total']) / book['count']