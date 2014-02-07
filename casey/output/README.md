
U (K x D), R (N x K), and ratings (N x D) are pickled files:

-	U is the K x D array of _user_ centroids
-	R is the N x K array of user responsibilities

-	ratings is the N x D sparse matrix mapping N users to their ratings of D books. ratings is represented as an LIL matrix (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix). To do any calculations on it, you'll need to convert it to CSR or CSC format (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix). See http://docs.scipy.org/doc/scipy/reference/sparse.html for an overview of sparse matrix types in SciPy.