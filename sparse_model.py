import torch
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric
from kmeans_pytorch import kmeans


class EGAE(torch.nn.Module):
    """
    X: n * d
    """
    def __init__(self, X, A, labels, alpha, layers=None, acts=None, max_epoch=10, max_iter=50,
                 learning_rate=10**-2, coeff_reg=10**-3,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(EGAE, self).__init__()
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency_sparse = get_adjacency_with_self_loops(A).tocsr()
        self.adjacency_tensor = to_sparse_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)
        self.n_clusters = self.labels.unique().shape[0]
        self.alpha = alpha
        if layers is None:
            layers = [32, 16]
        self.layers = layers
        if acts is None:
            layers_count = len(self.layers)
            acts = [torch.nn.functional.relu] * (layers_count - 1)
            acts.append(torch.nn.functional.linear)
        self.acts = acts
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg

        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]
        self.indicator = None
        self.embedding = self.X
        self._build_up()
        self.to(self.device)

    def _build_up(self):
        self._gcn_parameters = []
        layers_count = len(self.layers)
        for i in range(layers_count):
            if i is 0:
                self._gcn_parameters.append(get_weight_initial([self.input_dim, self.layers[i]]))
                continue
            self._gcn_parameters.append(get_weight_initial([self.layers[i - 1], self.layers[i]]))
        self._gcn_parameters = torch.nn.ParameterList(self._gcn_parameters)

    def update_embedding(self, embedding):
        self.embedding = embedding

    def forward(self, Laplacian):
        layers_count = len(self.layers)
        embedding = self.X
        for i in range(layers_count):
            embedding = Laplacian.mm(embedding.matmul(self._gcn_parameters[i]))
            if self.acts[i] is None:
                continue
            embedding = self.acts[i](embedding)
        epsilon = torch.tensor(10**-7).to(self.device)
        embedding = embedding / embedding.norm(dim=1).reshape((self.data_size, -1)).max(epsilon)
        return embedding

    def build_loss_reg(self):
        layers_count = len(self.layers)
        loss_reg = 0
        for i in range(layers_count):
            loss_reg += self._gcn_parameters[i].abs().sum()
            # loss_reg += self._gcn_parameters[i].norm()**2
        return loss_reg

    def build_loss_cross_entropy(self, embedding, ind):
        data_size = len(ind)
        ground_truth = self.adjacency_sparse[ind, :].todense()
        ground_truth = to_tensor(ground_truth).to(self.device)
        edge_count = ground_truth.sum()
        pos_weight = (self.data_size * data_size - edge_count) / edge_count
        epsilon = torch.tensor(10**-7).to(self.device)
        embedding_sub = embedding[ind, :]
        recons_A = embedding_sub.matmul(embedding.t())
        loss = pos_weight * ground_truth.mul((1 / torch.max(recons_A, epsilon)).log()) + \
                    (1 - ground_truth).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        return loss.sum() / (self.data_size * data_size)

    def build_loss(self, embedding, ind):
        # diagonal elements
        loss_1 = 0
        loss_2 = embedding.t() - embedding.t().matmul(self.indicator).matmul(self.indicator.t())
        loss_2 = loss_2.norm()**2 / (loss_2.shape[0] * loss_2.shape[1])

        loss_reg = self.build_loss_reg()
        loss = loss_1 + self.alpha * loss_2 + self.coeff_reg * loss_reg
        return loss

    def update_indicator(self, embedding=None):
        features = embedding if embedding is not None else self.embedding
        if features.requires_grad:
            features = features.detach()
        try:
            U, _, __ = torch.svd(features)
        except:
            print('SVD Not Convergence')
        self.indicator = U[:, :self.n_clusters]  # c-top
        self.indicator = self.indicator.detach()
        # return indicator

    def clustering(self):
        epsilon = torch.tensor(10**-7).to(self.device)
        indicator = self.indicator / self.indicator.norm(dim=1).reshape((self.data_size, -1)).max(epsilon)
        indicator = indicator.detach()
        prediction, _ = kmeans(indicator, self.n_clusters, device=self.device)
        prediction = prediction + 1
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction.cpu().numpy())
        return acc, nmi, ari, f1

    def run(self):
        self.update_indicator()
        acc, nmi, ari, f1 = self.clustering()
        print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        Laplacian = get_Laplacian(self.adjacency_sparse).to(self.device)
        for epoch in range(self.max_epoch):
            assert not self.indicator.requires_grad
            for i in range(self.max_iter):
                batch_size = 2000
                batch_count = int(np.ceil(self.data_size / batch_size))
                for j in range(batch_count):
                    if j != (batch_count-1):
                        ind = list(range(j*batch_size, min((j+1) * batch_size, self.data_size)))
                    else:
                        ind = list(range(self.data_size - batch_size, self.data_size))
                    optimizer.zero_grad()
                    embedding = self(Laplacian)
                    loss = self.build_loss(embedding, ind)
                    loss.backward()
                    optimizer.step()
                    objs.append(loss.item())
                # print('loss: ', loss.item())
            self.update_embedding(embedding)
            self.update_indicator()
            acc, nmi, ari, f1 = self.clustering()
            loss = self.build_loss(embedding, ind)
            objs.append(loss.item())
            print('loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))
        return np.array(objs)

    def build_pretrain_loss(self, embedding, ind):
        loss = self.build_loss_cross_entropy(embedding, ind)
        loss_reg = self.build_loss_reg()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = get_Laplacian(self.adjacency_sparse).to(self.device)
        embedding = None
        for i in range(pretrain_iter):
            batch_size = 2000
            batch_count = int(np.ceil(self.data_size / batch_size))
            for j in range(batch_count):
                if j != (batch_count - 1):
                    ind = list(range(j * batch_size, min((j + 1) * batch_size, self.data_size)))
                else:
                    ind = list(range(self.data_size - batch_size, self.data_size))
                optimizer.zero_grad()
                embedding = self(Laplacian)
                loss = self.build_pretrain_loss(embedding, ind)
                loss.backward()
                optimizer.step()
            # print(loss.item())
        print(loss.item())
        assert embedding is not None
        self.update_embedding(embedding)


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)


def to_sparse_tensor(X):
    return torch.sparse_coo_tensor(torch.LongTensor([X.row.tolist(), X.col.tolist()]),
                                   torch.FloatTensor(X.data.astype(np.float)))


def get_adjacency_with_self_loops(A):
    dim = A.shape[0]
    diag_indeces = (list(range(dim)), list(range(dim)))
    I = sp.coo_matrix((np.ones(dim), diag_indeces), shape=(dim, dim), dtype=np.float).tocsr()
    diag = sp.coo_matrix((A.diagonal(), diag_indeces), shape=(dim, dim), dtype=np.float).tocsr()
    L = A - diag + I
    return L.tocoo()


def get_Laplacian(A):
    # A should be a csr matrix
    dim = A.shape[0]
    diag_indeces = (list(range(dim)), list(range(dim)))
    I = sp.coo_matrix((np.ones(dim), diag_indeces), shape=(dim, dim), dtype=np.float).tocsr()
    diag = sp.coo_matrix((A.diagonal(), diag_indeces), shape=(dim, dim), dtype=np.float).tocsr()
    L = A - diag + I
    D = np.array(L.sum(1)).reshape(dim)
    sqrt_D = np.power(D, -1/2)
    sqrt_D = sp.coo_matrix((sqrt_D, diag_indeces), shape=(dim, dim), dtype=np.float).tocsr()
    Laplacian = sqrt_D.dot(L).dot(sqrt_D).tocoo()
    return to_sparse_tensor(Laplacian)


if __name__ == '__main__':
    testX = torch.rand(10, 4)
    gae = EGAE(testX, 0, 0, 0)
    for name, param in gae.named_parameters():
        print(name)
    # print(len(gae.parameters()))
