import torch
import numpy as np
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric


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
        self.adjacency = to_tensor(A).to(self.device)
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

    def forward(self, Laplacian):
        layers_count = len(self.layers)
        embedding = self.X
        for i in range(layers_count):
            embedding = Laplacian.mm(embedding.matmul(self._gcn_parameters[i]))
            if self.acts[i] is None:
                continue
            embedding = self.acts[i](embedding)
        epsilon = torch.tensor(10**-7).to(self.device)
        self.embedding = embedding / embedding.norm(dim=1).reshape((self.data_size, -1)).max(epsilon)

        recons_A = self.embedding.matmul(self.embedding.t())
        return recons_A

    def build_loss_reg(self):
        layers_count = len(self.layers)
        loss_reg = 0
        for i in range(layers_count):
            # LASSO
            loss_reg += self._gcn_parameters[i].abs().sum()
            # RIDGE
            # loss_reg += self._gcn_parameters[i].norm()**2
        return loss_reg

    def build_loss(self, recons_A):
        # diagonal elements
        epsilon = torch.tensor(10**-7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss_1 = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + \
                 (1 - self.adjacency).mul((1 / torch.max((1-recons_A), epsilon)).log())
        loss_1 = loss_1.sum() / (self.data_size**2)

        loss_2 = self.embedding.t() - self.embedding.t().matmul(self.indicator).matmul(self.indicator.t())
        loss_2 = loss_2.norm()**2 / (loss_2.shape[0] * loss_2.shape[1])

        loss_reg = self.build_loss_reg()
        loss = loss_1 + self.alpha * loss_2 + self.coeff_reg * loss_reg
        return loss

    def update_indicator(self, features):
        if features.requires_grad:
            features = features.detach()
        try:
            U, _, __ = torch.svd(features)
        except:
            print('SVD Not Convergence')
        self.indicator = U[:, :self.n_clusters]  # c-top
        self.indicator = self.indicator.detach()

    def clustering(self):
        epsilon = torch.tensor(10**-7).to(self.device)
        indicator = self.indicator / self.indicator.norm(dim=1).reshape((self.data_size, -1)).max(epsilon)
        indicator = indicator.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters).fit(indicator)
        prediction = km.predict(indicator)
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)
        return acc, nmi, ari, f1

    def run(self):
        self.update_indicator(self.embedding)
        acc, nmi, ari, f1 = self.clustering()
        print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for epoch in range(self.max_epoch):
            assert not self.indicator.requires_grad
            for i in range(self.max_iter):
                optimizer.zero_grad()
                recons_A = self(Laplacian)
                loss = self.build_loss(recons_A)
                loss.backward()
                optimizer.step()
                objs.append(loss.item())
                # print('loss: ', loss.item())
            self.update_indicator(self.embedding)
            acc, nmi, ari, f1 = self.clustering()
            loss = self.build_loss(recons_A)
            objs.append(loss.item())
            print('loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))
        return np.array(objs)

    def build_pretrain_loss(self, recons_A):
        epsilon = torch.tensor(10 ** -7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + \
                 (1 - self.adjacency).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        loss = loss.sum() / (loss.shape[0] * loss.shape[1])
        loss_reg = self.build_loss_reg()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
            # print(loss.item())
        print(loss.item())


class GAE(torch.nn.Module):
    """
    X: n * d
    """
    def __init__(self, X, A, labels, layers=None, acts=None, max_iter=200,
                 learning_rate=10**-2, coeff_reg=10**-3,
                 device=torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')):
        super(GAE, self).__init__()
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)
        self.n_clusters = self.labels.unique().shape[0]
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
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg

        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]
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

    def forward(self, Laplacian):
        layers_count = len(self.layers)
        embedding = self.X
        for i in range(layers_count):
            embedding = Laplacian.mm(embedding.matmul(self._gcn_parameters[i]))
            if self.acts[i] is None:
                continue
            embedding = self.acts[i](embedding)
        self.embedding = embedding

        recons_A = self.embedding.matmul(self.embedding.t())
        recons_A = recons_A.sigmoid()
        return recons_A

    def build_loss_reg(self):
        layers_count = len(self.layers)
        loss_reg = 0
        for i in range(layers_count):
            loss_reg += self._gcn_parameters[i].abs().sum()
            # loss_reg += self._gcn_parameters[i].norm()**2
        return loss_reg

    def build_loss(self, recons_A):
        # diagonal elements
        epsilon = torch.tensor(10**-7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss_1 = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + \
                 (1 - self.adjacency).mul((1 / torch.max((1-recons_A), epsilon)).log())
        loss_1 = loss_1.sum() / (self.data_size**2)

        loss_reg = self.build_loss_reg()
        loss = loss_1 + self.coeff_reg * loss_reg
        return loss

    def clustering(self):
        embedding = self.embedding.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)
        return acc, nmi, ari, f1

    def run(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_loss(recons_A)
            loss.backward()
            optimizer.step()
            # print('loss: ', loss.item())
        acc, nmi, ari, f1 = self.clustering()
        print('loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))


def get_weight_initial(shape):
    bound = np.sqrt(6.0 / (shape[0] + shape[1]))
    ini = torch.rand(shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)


def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1/2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian


if __name__ == '__main__':
    testX = torch.rand(10, 4)
    gae = EGAE(testX, 0, 0, 0)
    for name, param in gae.named_parameters():
        print(name)
