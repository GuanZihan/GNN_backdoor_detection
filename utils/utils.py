from typing import List, Tuple

import torch_geometric
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import copy
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_dense_adj
import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen import eigsh
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from src.models import GIN, GCN, GraphSAGE, NodeGCN


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        # print( data.x.shape )
        return data


def prepare_synthetic_dataset(dataset):
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)

            data.x = F.one_hot(degs.to(torch.int64), num_classes=max_degree+1).to(torch.float)
            print(data.x.shape)


        return dataset


def fgw_barycenter(aligned_graphs: List[np.ndarray],
                   aligned_ps: List[np.ndarray],
                   p_b: np.ndarray,
                   ws: np.ndarray,
                   inner_iters: int,
                   outer_iters: int,
                   beta: float,
                   gamma: float) -> np.ndarray:
    """
    Calculate smoothed Gromov-Wasserstein barycenter

    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param aligned_ps: a list of (Ni, 1) distributions
    :param p_b: (Nb, 1) distribution
    :param ws: (K, ) weights
    :param inner_iters: the number of sinkhorn iterations
    :param outer_iters: the number of barycenter iterations
    :param beta: the weight of proximal term
    :param gamma: the weight of gw term
    :return:
    """
    cost_ps = []
    trans = []
    priors = []
    for p in aligned_ps:
        cost_p = p_b ** 2 + (p ** 2).T - 2 * (p_b @ p.T)
        cost_p /= np.max(cost_p)
        cost_ps.append(cost_p)
        tran = proximal_ot(cost_p, p_b, p, iters=inner_iters, beta=beta)
        trans.append(tran)
        priors.append(tran + 1e-16)

    barycenter = None
    for o in range(outer_iters):
        # update smoothed barycenter
        averaged_graph = averaging_graphs(aligned_graphs, trans, ws)
        barycenter = averaged_graph / (p_b @ p_b.T)

        # update optimal transports
        for i in range(len(aligned_graphs)):
            cost_i = gw_cost(barycenter, aligned_graphs[i], trans[i], p_b, aligned_ps[i])
            cost_i = gamma * cost_i + (1 - gamma) * cost_ps[i]
            cost_i /= np.max(cost_i)
            trans[i] = proximal_ot(cost_i, p_b, aligned_ps[i], iters=inner_iters, beta=beta, prior=priors[i])

    barycenter[barycenter > 1] = 1
    barycenter[barycenter < 0] = 0
    return barycenter

def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)

def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st

def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph

def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float=1e-10,
                prior: np.ndarray=None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans

def prepare_dataset(dataset):
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset




def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()

def align_graphs_with_resuolution(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees
    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num


def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        # max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        # if N:
        #     aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
        #     normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num



def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        node_x = copy.deepcopy( node_x )
        sorted_node_x = node_x[ idx, :]

        max_num = max(max_num, N)
        # if max_num < N:
        #     max_num = max(max_num, N)
        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            # added
            aligned_node_x = np.zeros((max_num, 1))
            aligned_node_x[:num_i, :] = sorted_node_x


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            #added
            aligned_node_x = aligned_node_x[:N]

    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num


def two_graphons_mixup(two_graphons, la=0.5, num_sample=20):
    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):
        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)

        # print(edge_index)
    return sample_graphs



def two_x_graphons_mixup(two_x_graphons, la=0.5, num_sample=20):

    label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
    new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
    new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    sample_graph_x = torch.from_numpy(new_x).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.x = sample_graph_x
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes
        sample_graphs.append(pyg_graph)
        
        # print(edge_index)
    return sample_graphs



def graphon_mixup(dataset, la=0.5, num_sample=20):
    graphons = estimate_graphon(dataset, universal_svd)

    two_graphons = random.sample(graphons, 2)
    # for label, graphon in two_graphons:
    #     print( label, graphon )
    # print(two_graphons[0][0])

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    print("new label:", label)
    # print("new graphon:", new_graphon)

    # print( label )
    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) < new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]

        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        # print(sample_graph.shape)

        # print(sample_graph)

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)
        # print(edge_index)
    return sample_graphs


def estimate_graphon(dataset, graphon_estimator):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    # print(len(all_graphs_list))

    graphons = []
    for class_label in set(y_list):
        c_graph_list = [ all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label ]

        aligned_adj_list, normalized_node_degrees, max_num, min_num = align_graphs(c_graph_list, padding=True, N=400)

        graphon_c = graphon_estimator(aligned_adj_list, threshold=0.2)

        graphons.append((np.array(class_label), graphon_c))

    return graphons



def estimate_one_graphon(aligned_adj_list: List[np.ndarray], method="universal_svd"):

    if method == "universal_svd":
        graphon = universal_svd(aligned_adj_list, threshold=0.2)
    else:
        graphon = universal_svd(aligned_adj_list, threshold=0.2)

    return graphon



def split_class_x_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    all_node_x_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)
        all_node_x_list = [graph.x.numpy()]

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list, c_node_x_list ) )

    return class_graphs


def split_class_graphs(dataset):

    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()

        all_graphs_list.append(adj)

    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list ) )

    return class_graphs




def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon models
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    torch.cuda.empty_cache()
    return graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method

    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon


def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(  graph.edge_index.shape[1] )
    avg_num_nodes = sum( num_total_nodes ) / len(graphs_list)
    avg_num_edges = sum( num_total_edges ) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median( num_total_nodes ) 
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density


def show_graphs(dataset, idx):
    g = torch_geometric.utils.to_networkx(dataset[idx], to_undirected=True)
    np.random.seed(2021)
    random_select_nodes_1 = np.random.choice(len(g.nodes), 14, replace=False)
    print(random_select_nodes_1)

    color_map_1 = []
    for node in g:
        print(node)
        if node in random_select_nodes_1:
            color_map_1.append('orange')
        else:
            color_map_1.append('lightblue')
    plt.figure()
    nx.draw(g,  node_color=color_map_1, with_labels = True)
    plt.savefig("Test Clean")
    print(dataset[idx].y)
    print("Clean")


def get_data(dataset, idx, batch_size=1, shuffle = False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    for batch_idx, data in enumerate(loader):
        if not batch_idx == idx:
            continue
        return data

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj, norm=False):
    """Preprocessing of adjacency matrix for simple GCN models and conversion to tuple representation."""
    if norm:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sparse_to_tuple(adj_normalized)
    else:
        return sparse_to_tuple(sp.coo_matrix(adj))

def get_node_set(sub_edge_label_matrix):
    node_set = set()
    for e in preprocess_adj(sub_edge_label_matrix)[0][np.where(preprocess_adj(sub_edge_label_matrix)[1] == 1)]:
        node_set.add(e[0])
        node_set.add(e[1])
    return node_set

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def get_pretrained_model(args, num_features, PATH):
    model = GIN(num_features=num_features, num_classes=args.num_classes, num_hidden=args.num_hidden)
    data = torch.load(PATH)
    print(data)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

