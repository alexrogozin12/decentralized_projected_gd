import networkx as nx
import numpy as np
import matplotlib
import torch
import scipy
import torch.nn.functional as F
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from tqdm import tqdm_notebook


def _DEVICE():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu:0'


def get_graph_generator(n):
    def generator():
        return nx.generators.connected_watts_strogatz_graph(n, 4, 0.1)
    return generator


def count_laplacian(graph):
    lapl = -torch.DoubleTensor(nx.adj_matrix(graph).toarray())
    nodenums = torch.arange(graph.number_of_nodes())
    degrees = torch.DoubleTensor([graph.degree[key.item()] for key in nodenums])
    lapl[nodenums, nodenums] = degrees
    return lapl.to(_DEVICE())


def count_metropolis_laplacian(graph):
    n = graph.number_of_nodes()
    degrees = np.array([nx.degree(graph)[i] for i in range(n)] * n).reshape((n, n))
    maxs = np.max(np.vstack((degrees[None, :, :], degrees.T[None, :, :])), axis=0)

    res = nx.adj_matrix(graph).toarray() / (1. + maxs)
    nodenums = np.array(range(n))
    res[nodenums, nodenums] = 1. - res.sum(axis=0)
    return torch.DoubleTensor(res).to(_DEVICE())


def decentralized_logistic_regr(X, A_batches):
    pred_batches = []
    for x, A in zip(X.T, A_batches):
        pred_batches.append(torch.mm(A, x.view(-1, 1)))

    return torch.sigmoid(torch.cat(pred_batches)).flatten()


def project(Y, gen_graph, R=10., eps=1e-1, N_iter=100, gamma=1e-4, 
            return_hist=False):
    def proj_loss(X, W):
        return ((X - Y)**2).sum() / 2. + R**2 / eps * torch.sum(X * torch.mm(X, W))
    
    X = Y.clone().detach()
    Y_proj = Y.mean(dim=1).view(-1, 1)
    X.requires_grad_(True)
    r_k = []
    for step in range(N_iter):
        if X.grad is not None:
            X.grad.detach_()
            X.grad.zero_()
        graph = gen_graph()
        W = count_laplacian(graph)
        loss = proj_loss(X, W)
        loss.backward()
        X.data.add_(-gamma, X.grad.data)
        r_k.append(((X - Y_proj)**2).sum().item())
    
    if return_hist:
        return X, r_k
    else:
        return X


def projected_gradient(X_starting, A_batches, c_train, regcoef=0.01, 
                       gamma_outer=5., gamma_inner=1e-4, N_outer=200, N_inner=100, R=10., eps=0.1):
    
    hist = defaultdict(list)
    X = X_starting.clone().detach().requires_grad_(True)
    for step in tqdm_notebook(range(N_outer)):
        if X.grad is not None:
            X.grad.detach_()
            X.grad.zero_()
        res = decentralized_logistic_regr(X, A_batches)
        loss = torch.nn.BCELoss(reduction='mean')(res, c_train)
        loss.backward()
        hist['func'].append(loss.item())

        X_proj = X.mean(dim=1).view(-1, 1)
        hist['dist_from_k'].append(((X - X_proj)**2).sum().item())

        X.data.add_(-gamma_outer, X.grad.data)

        X.data = project(X, get_graph_generator(X.shape[1]), R, eps, 
                         N_inner, gamma_inner).data
    
    hist['X'] = X.data.clone().detach()
    return hist


def acc_projected_gradient(X_starting, A_batches, c_train, regcoef=0.01, momentum=0.99, 
                           gamma_outer=5., gamma_inner=1e-4, N_outer=200, N_inner=100, 
                           R=10., eps=0.1):
    
    hist = defaultdict(list)
    X = X_starting.clone().detach().requires_grad_(True)
    Y = X_starting.clone().detach().requires_grad_(False)
    Y_old = X_starting.clone().detach().requires_grad_(False)
    for step in tqdm_notebook(range(N_outer)):
        if X.grad is not None:
            X.grad.detach_()
            X.grad.zero_()
        res = decentralized_logistic_regr(X, A_batches)
        loss = torch.nn.BCELoss(reduction='mean')(res, c_train)
        loss.backward()
        hist['func'].append(loss.item())

        X_proj = X.mean(dim=1).view(-1, 1)
        hist['dist_from_k'].append(((X - X_proj)**2).sum().item())

        Y.data = X.data.clone().detach()
        Y.data.add_(-gamma_outer, X.grad.data)
        Y.data = project(Y, get_graph_generator(X.shape[1]), R, eps, 
                         N_inner, gamma_inner)
        X.data = Y.data.clone().detach()
        X.data.add_(momentum, (Y - Y_old).data)
        Y_old.data = Y.data.clone().detach()
    
    hist['X_star'] = X.data.clone().detach()
    return hist


def DIGing(X_starting, A_batches, c_train, N_iter=1000, alpha=0.5):
    X = X_starting.clone().detach().requires_grad_(True)
    res = decentralized_logistic_regr(X, A_batches)
    loss = torch.nn.BCELoss(reduction='mean')(res, c_train)
    loss.backward()
    Y = X.grad.clone().detach()
    old_grad = X.grad.clone().detach()
    gen_graph = get_graph_generator(X_starting.shape[1])
    hist = defaultdict(list)

    for step in tqdm_notebook(range(N_iter)):
        if X.grad is not None:
            X.grad.detach_()
            X.grad.zero_()

        W = count_metropolis_laplacian(gen_graph())

        res = decentralized_logistic_regr(X, A_batches)
        loss = torch.nn.BCELoss(reduction='mean')(res, c_train)
        loss.backward()
        hist['func'].append(loss.item())

        X_proj = X.mean(dim=1).view(-1, 1)
        hist['dist_from_k'].append(((X - X_proj)**2).sum().item())
        
        if Y is None:
            print('Initializing Y')
            Y = X.grad.clone().detach()
    
        X.data = torch.mm(X.data, W.data) - alpha * Y.data
    #     Y.data = torch.mm(Y.data, W.data) + n * (X.grad.data - old_grad)
        Y.data = torch.mm(Y.data, W.data) + (X.grad.data - old_grad)

        old_grad = X.grad.clone().detach()
    
    hist['X_star'] = X.data.clone().detach()
    return hist
