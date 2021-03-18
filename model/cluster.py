import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from evaluator.metrics import clustering_metrics
from evaluator.dist_modularity import cal_modularity


def commit_k_mean(embedding, origin_feature, adj_sp, true_label, cluster_k, params):
    """
    Execute K-means clustering and evaluate the performance as AGC

    :param embedding: learnt representation of nodes
    :param origin_feature: original node feature
    :param adj_sp: sparse adjacency of graph
    :param true_label: ground truth label for each node
    :param cluster_k: nb. of candidate classes

    :return: training log
    """

    rep = 10
    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)
    dm = np.zeros(rep)

    u, _, _ = sp.linalg.svds(embedding, k=cluster_k, which='LM')

    # Average value among several runs
    for i in range(rep):
        kmeans = KMeans(n_clusters=cluster_k).fit(u)
        predict_labels = kmeans.predict(u)
        cm = clustering_metrics(true_label, predict_labels)
        ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()
        dm[i] = cal_modularity(adj_sp, predict_labels, origin_feature, epsilon_=params['epsilon'])

    acc_means = np.mean(ac) * 100
    nmi_means = np.mean(nm) * 100
    f1_means = np.mean(f1) * 100
    dm_means = np.mean(dm) * 100

    msg = '|acc_mean:%.4f|nmi_mean:%.4f|f1_mean:%.4f|dm_mean:%.4f|' \
          % (acc_means, nmi_means, f1_means, dm_means)

    return msg
