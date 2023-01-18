# coding=utf-8
import sys

sys.path.append("/home/mengxuan/g-mixup/")
print(sys.path)
import logging
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import PGExplainer
import torch_geometric
import random
import argparse
import warnings
import networkx as nx
from scipy import sparse
warnings.filterwarnings('ignore')
from sklearn.decomposition import FastICA

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models import GIN, GCN, GraphSAGE, NodeGCN, GCN_N
from data_loader import inject_sub_trigger, S2VGraph, inject_explainability_graph
from graphcnn import GraphCNN
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer
from subgraphx import SubgraphX, find_closest_node_result, PlotUtils
from src.graphon_estimator import estimate_target_distribution
from utils.utils import split_class_graphs, stat_graph, align_graphs, universal_svd, \
    align_graphs_with_resuolution, fgw_barycenter
from degree import hier_explain
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def prepare_dataset_x(dataset, max_degree=0):
    # if dataset[0].x is None:
    if max_degree == 0:
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            data.num_nodes = int(torch.max(data.edge_index)) + 1
        max_degree = max_degree + 4  # edit this!!!!!!!!!!!!!!!!!!!!!!!!!1
    else:
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            data.num_nodes = int(torch.max(data.edge_index)) + 1
        max_degree = max_degree
    if max_degree < 10000:
        # dataset.transform = T.OneHotDegree(max_degree)

        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
    else:
        deg = torch.cat(degs, dim=0).to(torch.float)
        mean, std = deg.mean().item(), deg.std().item()
        for data in dataset:
            degs = degree(data.edge_index[0], dtype=torch.long)
            data.x = ((degs - mean) / std).view(-1, 1)
    return dataset, max_degree


def prepare_dataset_onehot_y(dataset, num_classes=2):
    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))

    for idx, data in enumerate(dataset):
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]

    return dataset


def train(args, model, criterion, train_loader, epoch):
    model.train()
    loss_all = 0
    correct = 0
    graph_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y
        loss = criterion(output, y).to(device)
        # loss = torch.sign(loss - 0.5) * loss

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all

    if epoch % 10 == 0:
        logger.info(
            'Epoch: {:03d}, Train Loss: {:.6f}'.format(
                epoch, loss))
    return model, loss


def test(args, model, criterion, loader, epoch, type="Train"):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]

        y = data.y
        loss += criterion(output, y).item() * data.num_graphs
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total

    if epoch % 5 == 0:
        logger.info(
            '[{}] Epoch: {:03d}, Test Loss: {:.6f} Test Accuracy {:.6f}'.format(str(type),
                                                                                epoch, loss, acc))
    return acc, loss


def preprocess_dataset(dataset, num_classes=2, max_degree=0):
    for graph in dataset:
        graph.y = graph.y.view(-1)

    # dataset = prepare_dataset_onehot_y(dataset, num_classes=num_classes)
    dataset, max_degree = prepare_dataset_x(dataset, max_degree=max_degree)
    return dataset, max_degree



def calculate_embedding_graph(args, model, loader):
    model.eval()
    embeddings = []
    labels = []
    for data in loader:
        def grad_hook(grad, inp, output):
            output = output.view(-1)
            embeddings.append(output.detach().cpu().numpy())

        if args.model in ["GraphCNN"]:
            handle = model.linears_prediction[3].register_forward_hook(grad_hook)
            batch_graph = [data]
            output = model(batch_graph)
            labels.append(data.label)
        else:
            data = data.to(device)
            handle = model.fc1.register_forward_hook(grad_hook)
            model(data.x, data.edge_index, data.batch)
            y = data.y
            labels.append(y.cpu().numpy())
        handle.remove()

    return np.array(embeddings), labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="MUTAG")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="True")

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

    parser.add_argument("--injection_ratio", default=0.1, type=float,
                        help="the number of injected samples to the training dataset")
    parser.add_argument("--split_ratio", default=0.9, type=float, help="train/test split ratio")
    parser.add_argument("--num_classes", default=2, type=int, help="number of classes")
    parser.add_argument("--trigger_size", default=3, type=int, help="# of Nodes to be poisoned")
    parser.add_argument("--trigger_density", default=0.8, type=float, help="Density of Subgraph Triggers")
    parser.add_argument("--device", default="0", type=str, help="GPU device index")

    parser.add_argument("--target_label", default=1, type=int, help="Label to be attacked")

    parser.add_argument("--explain_method", default="subgraphx", help="use which explainable method")
    parser.add_argument("--attack_method", default="subgraph", help="use which attack method")

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    log_screen = eval(args.log_screen)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    model = args.model

    handler = logging.FileHandler(
        "./logs/Dataset{}+TriggerSize{}+TriggerDensity{}+Attack{}+abl.txt".format(args.dataset, args.trigger_size,
                                                                     args.trigger_density, args.attack_method),
        encoding='utf-8', mode='a')

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(handler)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)
    logger.info(stat_graph(dataset))
    random.Random(seed).shuffle(dataset)

    train_nums = int(len(dataset) * args.split_ratio)
    test_nums = int(len(dataset) * (1 - args.split_ratio))

    train_dataset = dataset[:train_nums]
    test_dataset = dataset[train_nums:]

    if args.attack_method == 'subgraph':
        train_dataset, injected_graph_idx, backdoor_train_dataset, clean_train_dataset = inject_sub_trigger(args,
                                                                                                            copy.deepcopy(
                                                                                                                train_dataset),
                                                                                                            inject_ratio=args.injection_ratio,
                                                                                                            target_label=args.target_label,
                                                                                                            backdoor_num=args.trigger_size,
                                                                                                            density=args.trigger_density)

        logger.info("# Train Dataset {}".format(len(train_dataset)))
        logger.info("# Backdoor Train Dataset {}".format(len(backdoor_train_dataset)))
        logger.info("# Clean Train Dataset {}".format(len(clean_train_dataset)))

        _, _, backdoor_test_dataset, _ = inject_sub_trigger(args, copy.deepcopy(test_dataset), inject_ratio=1,
                                                            target_label=args.target_label,
                                                            density=args.trigger_density,
                                                            backdoor_num=args.trigger_size)

        _, _, _, clean_test_dataset = inject_sub_trigger(args, copy.deepcopy(test_dataset), inject_ratio=0,
                                                         target_label=args.target_label)
    elif args.attack_method == "explain_attack":
        train_dataset, injected_graph_idx, backdoor_train_dataset, clean_train_dataset = inject_explainability_graph(
            args, copy.deepcopy(train_dataset),
            inject_ratio=args.injection_ratio,
            target_label=args.target_label,
            backdoor_num=args.trigger_size,
            density=args.trigger_density)

        _, _, backdoor_test_dataset, _ = inject_explainability_graph(args, copy.deepcopy(test_dataset), inject_ratio=1,
                                                            target_label=args.target_label,
                                                            density=args.trigger_density,
                                                            backdoor_num=args.trigger_size)

        _, _, _, clean_test_dataset = inject_explainability_graph(args, copy.deepcopy(test_dataset), inject_ratio=0,
                                                         target_label=args.target_label)
    elif args.attack_method == 'GTA':
        train_dataset = []
        backdoor_test_dataset = []
        fixed_trainset = np.load("GTA/GTA_train_dataset_{}_{}.npy".format(args.dataset, args.trigger_size), allow_pickle=True)
        injected_graph_idx = np.load("GTA/GTA_perm_index_{}_{}.npy".format(args.dataset, args.trigger_size), allow_pickle=True)

        fixed_testset = np.load("GTA/GTA_backdoor_test_dataset_{}_{}.npy".format(args.dataset, args.trigger_size), allow_pickle=True)
        injected_graph_idx_test = np.load("GTA/GTA_perm_index_test_{}_{}.npy".format(args.dataset, args.trigger_size), allow_pickle=True)

        # gta_model_root = "GTA/gcn_{}.pth".format(args.dataset)

        # model_gcn = GCN_N(4, args.num_classes, [64, 64, 64, 64, 64], dropout=0)
        # model_gcn.load_state_dict(torch.load(gta_model_root))

        for idx, (graph, x, y) in enumerate(fixed_trainset):
            sparse_graph = sparse.csr_matrix(graph)
            edge_index, edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(sparse_graph)
            data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([y]))
            train_dataset.append(data)

        for idx, (graph, x, y) in enumerate(fixed_testset):
            if idx in injected_graph_idx_test:
                sparse_graph = sparse.csr_matrix(graph)
                edge_index, edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(sparse_graph)
                data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([y]))
                backdoor_test_dataset.append(data)


        _, _, _, clean_test_dataset = inject_sub_trigger(args, copy.deepcopy(test_dataset), inject_ratio=0,
                                                         target_label=args.target_label)


    else:
        raise NotImplementedError



    logger.info("# Test Dataset {}".format(len(test_dataset)))
    logger.info("# Backdoor Test Dataset {}".format(len(backdoor_test_dataset)))
    logger.info("# Clean Test Dataset {}".format(len(clean_test_dataset)))

    # After splitting, preprocess all the dataset

    train_dataset, max_degree = preprocess_dataset(train_dataset, num_classes=args.num_classes)
    backdoor_test_dataset, _ = preprocess_dataset(backdoor_test_dataset, num_classes=args.num_classes,
                                                  max_degree=max_degree)
    clean_test_dataset, _ = preprocess_dataset(clean_test_dataset, num_classes=args.num_classes, max_degree=max_degree)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=batch_size)


    num_features = train_dataset[0].x.shape[1]

    ##################

    ##################

    # preprocess the batch data

    train_backdoor = []
    test_graphs = []
    test_backdoor = []

    if model == "GIN":
        model = GIN(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model == "GCN":
        model = GCN(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model == "sage":
        model = GraphSAGE(num_features=num_features, num_classes=args.num_classes, num_hidden=num_hidden).to(device)
    elif model == "NodeGCN":
        model = NodeGCN(num_features=num_features, num_classes=args.num_classes).to(device)
    elif model == "GraphCNN":
        # implemented by https://github.com/zaixizhang/graphbackdoor
        for graph in train_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            train_backdoor.append(graph_s2v)

        for graph in clean_test_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            test_graphs.append(graph_s2v)

        for graph in backdoor_test_dataset:
            net = torch_geometric.utils.to_networkx(graph)
            graph_s2v = S2VGraph(net, graph.y.item(), None, graph.x)
            graph_s2v.edge_mat = graph.edge_index
            test_backdoor.append(graph_s2v)

        model = GraphCNN(args.num_layers, args.num_mlp_layers, train_backdoor[0].node_features.shape[1],
                         args.hidden_dim, args.num_classes, args.final_dropout, args.lr, args.graph_pooling_type,
                         args.neighbor_pooling_type, device).to(device)

    else:
        logger.info(f"No models.")




    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    clean_losses = []
    bad_losses = []


    if args.model == "GraphCNN":
        pass
    else:
        for epoch in range(1, num_epochs):
            model, _ = train(args, model, criterion, train_loader, epoch)
            train_acc, _ = test(args, model, criterion, train_loader, epoch, type="Trainset")

            test_acc, test_loss = test(args, model, criterion, clean_test_loader, epoch, type="Clean Test")
            backdoor_test_acc, backdoor_test_loss = test(args, model, criterion, backdoor_test_loader, epoch,
                                                         type="Backdoor Test")


            scheduler.step()
            if epoch % 5 == 0:
                logger.info("=================Epoch {}=================".format(epoch))


    model.eval()
    all_losses = []
    for idx, graph in enumerate(train_dataset):
        output = model(graph.x.to(device), graph.edge_index.to(device))
        loss = criterion(output, graph.y.to(device))
        all_losses.append(loss.cpu().item())

    all_losses = np.array(all_losses)
    y_label = np.zeros((len(all_losses), 1))
    y_label[injected_graph_idx] = 1

    tps = []
    fps = []

    for i in sorted([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.4, 0.6, 0.8, 1], reverse=True):
        select_idx = np.argsort(all_losses)[:int(i * len(all_losses))]
        y_pred = np.zeros((len(all_losses), 1))
        y_pred[select_idx] = 1
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
        tps.append(tp / len(injected_graph_idx))
        fps.append(fp / (len(all_losses) - len(injected_graph_idx)))

        accuracy = accuracy_score(y_label, y_pred)

        logger.info("[{}] Accuracy Score {}".format(i, accuracy))


    auc = -1 * np.trapz(tps, fps)
    logger.info("FP {}".format(fps))
    logger.info("TP {}".format(tps))
    logger.info("AUC {}".format(auc))

    plt.plot(fps, tps, linestyle='--', marker='o', color='darkorange', lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f' % auc)
    plt.legend(loc="lower right")
    plt.savefig('AUC_example{}_{}_{}_{}.png'.format(args.dataset, args.trigger_size, args.trigger_density, args.seed))
    plt.show()