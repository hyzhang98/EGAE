import numpy as np
import scipy.sparse as sp
import scipy.io as scio


def load_cora():
    path = 'data/cora/'
    data_name = 'cora'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), idx_map, adj.toarray(), labels


def load_wiki():
    print('Loading from raw data file...')
    data = scio.loadmat('data/wiki.mat')
    adj = data['adj']
    adj = adj - np.diag(np.diag(adj))
    features = data['X']
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    return features, None, adj, labels


def load_pubmed():
    print('Loading from raw data file...')
    data = scio.loadmat('data/pubmed.mat')
    adj = data['W']
    features = data['fea']
    labels = data['gnd']
    labels = np.reshape(labels, (labels.shape[0],))
    return features, None, adj.tocoo(), labels


def load_citeseer():
    path = 'data/citeseer/'
    data_name = 'citeseer'
    print('Loading from raw data file...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, data_name), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(idx_features_labels[:, -1], return_index=True, return_inverse=True)

    idx = np.array(idx_features_labels[:, 0], dtype=np.str)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, data_name), dtype=np.str)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    rows_to_delete = []
    for i in range(edges_unordered.shape[0]):
        if edges[i, 0] is None or edges[i, 1] is None:
            rows_to_delete.append(i)
    edges = np.array(np.delete(edges, rows_to_delete, axis=0), dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), idx_map, adj.toarray(), labels


def load_data(name):
    if name.lower() == 'cora':
        features, _, adj, labels = load_cora()
        return features, adj, labels
    elif name.lower() == 'citeseer':
        features, _, adj, labels = load_citeseer()
        return features, adj, labels
    elif name.lower() == 'wiki':
        features, _, adj, labels = load_wiki()
        return features, adj, labels
    elif name.lower() == 'pubmed':
        features, _, adj, labels = load_pubmed()
        return features, adj, labels
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))
    adj = data['W']
    return data['X'], adj, labels


if __name__ == '__main__':
    load_data(YALE)
