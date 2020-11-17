import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def data_load(dataset):
    """
    Load data from input file

    :param dataset: name of dataset

    :return graph_filter: function used for graph convolution as AGC
    :return adj: adjacency matrix
    :return feature: initial feature of nodes
    :return true_label: ground truth label for nodes

    """

    data = sio.loadmat('data/{}.mat'.format(dataset))
    feature = data['fea']
    if sp.issparse(feature):
        feature = feature.todense()

    adj = data['W']
    true_label = data['gnd']
    true_label = true_label.T
    true_label = true_label - 1
    true_label = true_label[0, :]

    cluster_k = len(np.unique(true_label))

    adj_sp = sp.coo_matrix(adj)
    # D^-1/2 A D^-1/2 or D^-1 A
    adj_norm = preprocess_adj(adj_sp)
    # G = 1/2（I + D^-1/2 A D^-1/2）
    graph_filter = (sp.eye(adj_norm.shape[0]) + adj_norm) / 2

    return graph_filter, adj, feature.astype('float'), true_label, cluster_k


def normalize_adj(adj, type='sym'):
    """Totally same as AGC paper
    Symmetrically normalize adjacency matrix. Derived from github"""
    if type == 'sym':
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    elif type == 'rw':
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj_normalized = d_mat_inv.dot(adj)
        return adj_normalized


def preprocess_adj(adj, type='sym', loop=True):
    """Totally same as AGC paper
    Preprocessing of adjacency matrix for simple
    GCN model and conversion to tuple representation."""
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj, type=type)
    return adj_normalized
