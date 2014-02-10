def stoch_grad_desc(rs, f, qs, ps):
    

def mfact(R, K, steps=5000, alpha=0.0002, beta=0.02):
    """
    Adapted from Albert Au Yeung (2010)
    http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/

    Arguments:
        R     : a matrix to be factorized, dimension N x D
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter

    Returns:
        P     : an initial matrix of dimension N x K
        Q     : an initial matrix of dimension D x K
    """
    N = R.shape[0]
    D = R.shape[1]

    P = np.random.rand(N,K)
    Q = np.random.rand(D,K)
    Q = Q.T

    # Biases
    Bu = np.zeros((N,1)) # N x 1
    Bi = np.zeros((D,1)) # D x 1

    # mean = ?? # calculate mean of R


    debug("Starting Matrix Factorization into %d principal components...", (K,))

    for step in xrange(steps):
        for i in xrange(N):
            for j in xrange(D):
                if R[i,j] > 0:
                    eij = R[i,j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in xrange(N):
            for j in xrange(D):
                if R[i,j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        debug("Step %d / %d: e = %d", (step, steps,e))
        if e < 0.001:
            break
    return P, Q.T