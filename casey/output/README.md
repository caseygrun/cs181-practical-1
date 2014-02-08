
There are a few data files in this directory: U_K (K x D), R_K (N x K), and ratings (N x D) are pickled files:

-	U_K is the K x D array of _user_ centroids for K centroids
-	R is the N x K array of user responsibilities for K centroids

-	svd_K is a triple containing the three matrices of the K-singular value decomposition of `ratings`:
	-	u : ndarray, shape=(M, k)
		Unitary matrix having left singular vectors as columns.
	-	s : ndarray, shape=(k,)
		The singular values.
	-	vt : ndarray, shape=(k, N)
		Unitary matrix having right singular vectors as rows.

-	ratings is a dict containing two entries: 
	-	"mean" is the mean of the overall ratings
	-	"variance" is the variance of the overall ratings
	-	"book_isbn_to_index" is a dict mapping ISBNs to numerical indices
	-	"ratings" is the N x D sparse matrix mapping N users to their ratings of D books. ratings is represented as an LIL matrix (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix). To do any calculations on it, you'll need to convert it to CSR or CSC format (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix). See http://docs.scipy.org/doc/scipy/reference/sparse.html for an overview of sparse matrix types in SciPy.