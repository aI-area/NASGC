import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances


def cal_modularity(adj_, label_, feature_=None, similarity_function_=euclidean_distances, epsilon_=10.):
    """
    1. P = (d_i*d_j)/2m or similarity matrix
    2. A: adjacency matrix
    3. modularity matrix = A - P, sum(modularity matrix) = 0
    4. modularity score: [-1, 1]

    function:
        Modularity = 1/2m sum_ij (A_ij - (d_i*d_j)/2m)*cluster(c_i,c_j)

    :param adj_: the adjacency of graph dataset (not include Identity matrix)
    :param label_: graph true label or pre label
    :param feature_: graph original feature or graph embedding
    :param similarity_function_: similarity distant function (euclidean distances)
    :param epsilon_: for P_ij_Dist function, epsilon > 0
    """

    def build_cluster_matrix():
        """
        for Kronecker Function:

        build cluster matrix, shape = [n, n]
        if node_i and node_j in same cluster, cluster_ij = 1 else 0
        """
        k = len(np.unique(label_))
        l_ = np.zeros([label_.shape[0], k], dtype=np.float32)
        l_[range(label_.shape[0]), label_] = 1
        return l_.dot(l_.T)

    def build_P_distance_matrix():
        """
        1. similarity_function_: s_ii = 0, s_ij = s_ji (euclidean_distances)
        2. deterrence function = exp(-(s_ij/epsilon)^2) , (0,1]
        3. epsilon > 0
            (If epsilon is small, deterrence function decreases sharply, indicating a short-
            range field; If epsilon is large, deterrence function decreases slowly, indicating a
            long-range field)
        4. d_i means the degree of node i

        function:
            P_ij_Dist = P_ji_Dist
            sum(P_ij_Dist) = sum(adj) = 2m

            P_ij = d_i*d_j exp(-(s_ij/epsilon)^2) / sum_v d_v exp(-(s_iv/epsilon)^2)
            P_ij_Dist = (P_ij + P_ji)/2
        """

        deterrence_fun_ = np.exp(-1*(similarity_function_(feature_) / epsilon_)**2)
        assert (deterrence_fun_ <= 1.).all()

        di_dj_e_ = di_dj_ * deterrence_fun_   # [n, n]*[n, n]
        dv_e_ = degree_ * deterrence_fun_     # [n, 1]*[n, n]

        P_ij_ = di_dj_e_ / (dv_e_.sum(1)[:, np.newaxis]+1e-8)     # [n, n] / [n, 1]
        P_ij_Dist_ = (P_ij_ + P_ij_.T) / 2.
        assert (P_ij_Dist_ == P_ij_Dist_.T).all()   # Matrix symmetry

        return P_ij_Dist_

    adj_ = adj_.astype('float32')
    feature_ = feature_.astype('float32')
    if sp.issparse(adj_):
        adj_ = adj_.toarray()

    m2_ = adj_.sum()    # the number of graph edge
    cluster_= build_cluster_matrix()     # [n, n], if i and j in same cluster, cluster_ij = 1 else 0
    degree_ = adj_.sum(1)[:, np.newaxis] # [n, 1], the degree of adj
    di_dj_ = degree_.dot(degree_.T)      # [n, n], di*dj

    if feature_ is None:
        one_ = di_dj_ / m2_   # [n, n]
        modularity_matrix_ = adj_ - one_
        assert modularity_matrix_.sum() < 1e-6
    else:
        one_ = build_P_distance_matrix()
        # a = one_.sum()
        # assert one_.sum()-m2_ < 1.
        modularity_matrix_ = adj_ - one_

    modularity_score_ = 1 / m2_ * (modularity_matrix_*cluster_).sum()
    return modularity_score_
