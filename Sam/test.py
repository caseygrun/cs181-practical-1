import shared_utils as su
import numpy as np

d = su.unpickle("casey/output/ratings")
print d
ratings = d["ratings"]
N = d["N"]
D = d["D"]
book_isbn_to_index = d["book_isbn_to_index"]
