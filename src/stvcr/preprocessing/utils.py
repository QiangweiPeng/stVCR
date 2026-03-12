import numpy as np
import ot as pot
import scipy.sparse

def generalized_procrustes_analysis(X, Y, pi, output_params=False, matrix=True):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).
    [from PASTE package]

    Args:
        X: np array of spatial coordinates (ex: sliceA.obs['spatial'])
        Y: np array of spatial coordinates (ex: sliceB.obs['spatial'])
        pi: mapping between the two layers output by PASTE
        output_params: Boolean of whether to return rotation angle and translations along with spatial coordiantes.
        matrix: Boolean of whether to return the rotation as a matrix or an angle.


    Returns:
        Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y.
    """

    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    # print(np.linalg.det(R))

    if output_params and not matrix:
        M = np.array([[0, -1], [1, 0]])
        theta = np.arctan(np.trace(M.dot(H)) / np.trace(H))
        return X, Y, theta, tX, tY
    elif output_params and matrix:
        return X, Y, R, tX, tY
    else:
        return X, Y

def rigid_body_transformation_invariant_OT(sliceA, sliceB, down_sampling_number = 5000,alpha=0.002, iter_num=5, spatial_key='spatial'):
    # from paste.visualization import generalized_procrustes_analysis
    # from paste.helper import intersect, extract_data_matrix, to_dense_array, kl_divergence
    # common_genes = intersect(sliceA.var.index, sliceB.var.index)

    common_genes = sliceA.var.index.intersection(sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # down spamling for faster
    if sliceA.n_obs + sliceB.n_obs > down_sampling_number:
        sliceA = sliceA[np.arange(0, sliceA.n_obs, int((sliceA.n_obs + sliceB.n_obs) / down_sampling_number)), :]
        sliceB = sliceB[np.arange(0, sliceB.n_obs, int((sliceA.n_obs + sliceB.n_obs) / down_sampling_number)), :]

    # Calculate spatial distances
    coordinatesA = sliceA.obsm[spatial_key].copy()
    coordinatesB = sliceB.obsm[spatial_key].copy()
    coordinatesA = coordinatesA - np.mean(coordinatesA, axis=0)
    coordinatesB = coordinatesB - np.mean(coordinatesB, axis=0)

    # Calculate expression dissimilarity
    A_X = sliceA.X.A if scipy.sparse.issparse(sliceA.X) else sliceA.X
    B_X = sliceB.X.A if scipy.sparse.issparse(sliceB.X) else sliceB.X
    X_cost = pot.dist(A_X, B_X)

    # init distributions
    a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]

    # init transformation matrix
    R_out = np.eye(coordinatesA.shape[1])

    for iter in range(iter_num):
        if iter == 0:
            continue
            pi = pot.emd(a=a, b=b, M=X_cost)
            coordinatesA, coordinatesB, R, _, _ = generalized_procrustes_analysis(X=coordinatesA, Y=coordinatesB,
                                                                                     pi=pi,
                                                                                     output_params=True, matrix=True)
            R_out = np.matmul(R_out, R)
        else:
            coordinates_cost = pot.dist(coordinatesA, coordinatesB)
            cost = (1 - alpha) * X_cost + alpha * coordinates_cost
            pi = pot.emd(a=a, b=b, M=cost)
            coordinatesA, coordinatesB, R, _, _ = generalized_procrustes_analysis(X=coordinatesA, Y=coordinatesB, pi=pi, 
                                                                                  output_params=True, matrix=True)
            R_out = np.matmul(R_out, R)

    return pi, R_out