# import torch_geometric
import networkx as nx
import numpy as np
import torch
import sys
sys.path.append("GNN_backdoor_detection")
sys.path.append("..")
print(sys.path)
import matplotlib.pyplot as plt
# from Regal import xnetmf
import argparse
from utils.utils import get_pretrained_model
from torch_geometric.utils import to_dense_adj
from config_subgraph import Graph, RepMethod
from torch_geometric.nn import GNNExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL.")

    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt',
                        help="Edgelist of combined input graph")

    parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
                        help='Embeddings path')

    parser.add_argument('--attributes', nargs='?', default=None,
                        help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

    parser.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser.add_argument('--untillayer', type=int, default=6,
                        help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default=0.2, help="Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser.add_argument('--numtop', type=int, default=0,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    return parser.parse_args()


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0
        self.max_neighbor = 0


def inject_sub_trigger(args, dataset, mode="ER", inject_ratio=0.1, backdoor_num=4, target_label=1, density=0.8):
    """
    Inject a sub trigger into the clean graph, return the poisoned dataset
    :param inject_ratio:
    :param dataset:
    :param mode:
    :return:
    """
    if mode == "ER":
        G_gen = nx.erdos_renyi_graph(backdoor_num, density, seed=args.seed)
    else:
        raise NotImplementedError

    print("The edges in the generated subgraph ", G_gen.edges)



    possible_target_graphs = []

    for idx, graph in enumerate(dataset):
        if graph.y.item() != target_label:
            possible_target_graphs.append(idx)

    np.random.seed(args.seed)
    injected_graph_idx = np.random.choice(possible_target_graphs, int(inject_ratio * len(dataset)))

    backdoor_dataset = []
    clean_dataset =[]
    all_dataset = []

    for idx, graph in enumerate(dataset):
        if idx not in injected_graph_idx:
            all_dataset.append(graph)
            clean_dataset.append(graph)
            continue

        if graph.num_nodes > backdoor_num:
            np.random.seed(args.seed)
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num, replace=False)
        else:
            np.random.seed(args.seed)
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num)

        removed_index = []
        ls_edge_index = graph.edge_index.T.numpy().tolist()

        # remove existing edges between the selected nodes
        for row_idx, i in enumerate(random_select_nodes):
            for col_idx, j in enumerate(random_select_nodes):
                if [i, j] in ls_edge_index:
                    removed_index.append(ls_edge_index.index([i, j]))


        removed_index = list(set(removed_index))
        remaining_index = np.arange(0, len(graph.edge_index[0, :]))
        remaining_index = np.delete(remaining_index, removed_index)

        graph.edge_index = graph.edge_index[:, remaining_index]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[remaining_index, :]

        # inject subgraph trigger into the clean graph
        for edge in G_gen.edges:
            i, j = random_select_nodes[edge[0]], random_select_nodes[edge[1]]

            # injecting edge
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(i)], [int(j)]])), dim=1)
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(j)], [int(i)]])), dim=1)
            # padding for the edge attributes matrix
            if graph.edge_attr is not None:
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)), dim=0)
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)),
                    dim=0)
        graph.y = torch.Tensor([target_label]).to(torch.int64)
        backdoor_dataset.append(graph)
        all_dataset.append(graph)

    return all_dataset, list(set(injected_graph_idx)), backdoor_dataset, clean_dataset

def inject_explainability_graph(args, dataset, mode="ER", inject_ratio=0.1, backdoor_num=4, target_label=1, density=0.8):
    if mode == "ER":
        G_gen = nx.erdos_renyi_graph(backdoor_num, density, seed=args.seed)
    else:
        raise NotImplementedError

    print("The edges in the generated subgraph ", G_gen.edges)

    possible_target_graphs = []

    for idx, graph in enumerate(dataset):
        if graph.y.item() != target_label:
            possible_target_graphs.append(idx)

    np.random.seed(args.seed)
    injected_graph_idx = np.random.choice(possible_target_graphs, int(inject_ratio * len(dataset)))

    backdoor_dataset = []
    clean_dataset = []
    all_dataset = []

    node_matrix = np.load("node_matrix_MUTAG.npy", allow_pickle=True)

    for idx, graph in enumerate(dataset):
        if idx not in injected_graph_idx:
            all_dataset.append(graph)
            clean_dataset.append(graph)
            continue

        if graph.num_nodes > backdoor_num:
            np.random.seed(args.seed)

            random_select_nodes = node_matrix[idx % 90][:backdoor_num]
            # print(random_select_nodes)
            # input()
        else:
            np.random.seed(args.seed)
            random_select_nodes = np.random.choice(graph.num_nodes, backdoor_num)

        removed_index = []
        ls_edge_index = graph.edge_index.T.numpy().tolist()

        # remove existing edges between the selected nodes
        for row_idx, i in enumerate(random_select_nodes):
            for col_idx, j in enumerate(random_select_nodes):
                if [i, j] in ls_edge_index:
                    removed_index.append(ls_edge_index.index([i, j]))

        removed_index = list(set(removed_index))
        remaining_index = np.arange(0, len(graph.edge_index[0, :]))
        remaining_index = np.delete(remaining_index, removed_index)

        graph.edge_index = graph.edge_index[:, remaining_index]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[remaining_index, :]

        # inject subgraph trigger into the clean graph
        for edge in G_gen.edges:
            i, j = random_select_nodes[edge[0]], random_select_nodes[edge[1]]

            # injecting edge
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(i)], [int(j)]])), dim=1)
            graph.edge_index = torch.cat((graph.edge_index, torch.LongTensor([[int(j)], [int(i)]])), dim=1)
            # padding for the edge attributes matrix
            if graph.edge_attr is not None:
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)), dim=0)
                graph.edge_attr = torch.cat(
                    (graph.edge_attr, torch.unsqueeze(torch.zeros_like(graph.edge_attr[0, :]), 0)),
                    dim=0)
        graph.y = torch.Tensor([target_label]).to(torch.int64)
        backdoor_dataset.append(graph)
        all_dataset.append(graph)

    return all_dataset, list(set(injected_graph_idx)), backdoor_dataset, clean_dataset
