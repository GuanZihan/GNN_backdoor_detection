import networkx as nx
import numpy as np
import torch_geometric
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from GNNs_unsupervised import GNN
import argparse
import random
import logging
import torch
import os.path as osp
from torch_geometric.datasets import TUDataset
import copy
import matplotlib.pyplot as plt

from src.data_loader import inject_sub_trigger
from src.attack import preprocess_dataset

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def example_with_cora():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--models', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gmixup', type=str, default="False")
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--aug_ratio', type=float, default=0.15)
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="False")
    parser.add_argument('--ge', type=str, default="MC")

    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear models.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')

    parser.add_argument("--injection_ratio", default=0.1, type=float, help="the number of injected samples to the training dataset")
    parser.add_argument("--split_ratio", default=0.9, type=float, help="train/test split ratio")

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    ge = args.ge
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num
    model = args.model

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)
    random.seed(seed)
    random.shuffle(dataset)

    train_nums = int(len(dataset) * args.split_ratio)
    test_nums = int(len(dataset) * (1-args.split_ratio))

    train_dataset = dataset[:train_nums]
    test_dataset = dataset[train_nums:]

    train_dataset, injected_graph_idx, backdoor_train_dataset, clean_train_dataset = inject_sub_trigger(copy.deepcopy(train_dataset), inject_ratio=args.injection_ratio, target_label=0)
    logger.info("# Train Dataset {}".format(len(train_dataset)))
    logger.info("# Backdoor Train Dataset {}".format(len(backdoor_train_dataset)))
    logger.info("# Clean Train Dataset {}".format(len(clean_train_dataset)))
    _, _, backdoor_test_dataset, _ = inject_sub_trigger(copy.deepcopy(test_dataset), inject_ratio=1, target_label=0)
    _, _, _, clean_test_dataset = inject_sub_trigger(copy.deepcopy(test_dataset), inject_ratio=0, target_label=0)
    logger.info("# Test Dataset {}".format(len(test_dataset)))
    logger.info("# Backdoor Test Dataset {}".format(len(backdoor_test_dataset)))
    logger.info("# Clean Test Dataset {}".format(len(clean_test_dataset)))
    train_dataset, max_degree = preprocess_dataset(train_dataset, num_classes=3)
    backdoor_test_dataset, _ = preprocess_dataset(backdoor_test_dataset, num_classes=3, max_degree=max_degree)
    clean_test_dataset, _ = preprocess_dataset(clean_test_dataset, num_classes=3, max_degree=max_degree)

    for graph in train_dataset:
        adj_matrix = torch_geometric.utils.convert.to_scipy_sparse_matrix(graph.edge_index).tocsr()
        raw_features = graph.x.numpy()
        """
        Example of using Graph Attention Network for unsupervised learning.
        using CUDA and print training progress
        """
        gnn = GNN(adj_matrix, features=raw_features, supervised=False, model='gat', device='cuda')
        # train the models
        gnn.fit()
        # get the node embeddings with the trained GAT
        embs = gnn.generate_embeddings()

        g = torch_geometric.utils.to_networkx(graph)
        np.random.seed(2021)
        random_select_nodes_1 = np.random.choice(len(g.nodes), 5, replace=False)
        color_map_1 = []
        for node in g:
            print(node)
            if node in random_select_nodes_1:
                color_map_1.append('orange')
            else:
                color_map_1.append('lightblue')
        plt.figure()
        nx.draw(g,  node_color=color_map_1, with_labels = True)
        plt.savefig("Graph")
        print(random_select_nodes_1)
        print(embs[random_select_nodes_1,:])
        print(embs)
        input()

if __name__ == "__main__":
    example_with_cora()
