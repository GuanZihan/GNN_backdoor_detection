# coding=utf-8
import sys

from torch_geometric.explain import Explainer

sys.path.append("/home/mengxuan/g-mixup/")
sys.path.append("/home/myid/zg37632/WORK_SPACE/g-mixup/")
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
from torch_geometric.explain.algorithm import GNNExplainer
from subgraphx import SubgraphX, find_closest_node_result, PlotUtils
from src.graphon_estimator import estimate_target_distribution
from utils.utils import split_class_graphs, stat_graph, align_graphs, universal_svd, \
    align_graphs_with_resuolution, fgw_barycenter
from degree import hier_explain
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

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


def train_(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    logger.info(
        'Epoch: {:03d}, Train Loss: {:.6f}'.format(
            epoch, average_loss))

    return average_loss


###pass data to models with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test_(model, device, test_graphs, type="backdoor"):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)

    pred = output.max(1, keepdim=True)[1]

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    loss = criterion(output, labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    logger.info(
        '[{}] Epoch: {:03d}, Test Loss: {:.6f} Test Accuracy {:.6f}'.format(str(type),
                                                                            epoch, loss, acc_test))

    return acc_test


def plot_node_embedding(args, embeddings, labels, gid=0):
    emb = embeddings[gid]
    np.random.seed(2021)
    # print(emb.shape)
    random_select_nodes = np.random.choice(emb.shape[0], args.trigger_size, replace=False)

    tsne = manifold.TSNE(n_components=2, perplexity=5).fit_transform(embeddings[gid])
    color = []
    for i in range(emb.shape[0]):
        if i in random_select_nodes:
            color.append("red")
        else:
            color.append("black")
    plt.figure()
    plt.scatter(tsne[:, 0], tsne[:, 1], c=color)
    plt.savefig("visualize_tsne{}".format(gid))
    plt.show()


def plot_graph_embedding(embeddings, bacdkoor_gids, labels):
    tsne = manifold.TSNE(n_components=2, perplexity=5).fit_transform(embeddings)
    color = []
    label_unique = np.unique(np.array(labels))
    colors = ["blue", "yellow", "lightblue", "green", "orange", "grey"]
    for i in range(embeddings.shape[0]):
        if i in bacdkoor_gids:
            color.append("red")
        else:
            color.append(colors[np.where(label_unique == labels[i])[0][0]])
    plt.figure()
    plt.scatter(tsne[:, 0], tsne[:, 1], c=color)
    legend = ["class" + str(idx) for idx in label_unique]
    plt.legend(legend)
    plt.savefig("visualize_tsne_graph")
    plt.show()


def calculate_embedding(args, model, loader):
    model.eval()
    embeddings = []
    labels = []
    for data in loader:
        def grad_hook(grad, inp, output):
            # output = output.view(-1)
            embeddings.append(output.detach().cpu().numpy())

        if args.model in ["GraphCNN"]:
            handle = model.mlps[3].register_forward_hook(grad_hook)
            batch_graph = [data]
            output = model(batch_graph)
            labels.append(data.label)
        else:
            data = data.to(device)
            handle = model.conv5.register_forward_hook(grad_hook)
            model(data.x, data.edge_index, data.batch)
            y = data.y
            labels.append(y.cpu().numpy())
        handle.remove()
    return np.array(embeddings), labels


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
        "./logs/Dataset{}+TriggerSize{}+TriggerDensity{}+Attack{}_ExD.txt".format(args.dataset, args.trigger_size,
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
        # implemented by https://github.com/zaixizhang/graphbackdoor
        for epoch in range(1, args.epoch + 1):
            scheduler.step()
            avg_loss = train_(args, model, device, train_backdoor, optimizer, epoch)
            acc_train = test_(model, device, train_backdoor, type="Trainset")
            acc_test_clean = test_(model, device, test_graphs, type="Clean")
            acc_test_backdoor = test_(model, device, test_backdoor, type="Backdoor")
    else:
        for epoch in range(1, num_epochs):
            model, _ = train(args, model, criterion, train_loader, epoch)
            train_acc, _ = test(args, model, criterion, train_loader, epoch, type="Trainset")

            test_acc, test_loss = test(args, model, criterion, clean_test_loader, epoch, type="Clean Test")
            backdoor_test_acc, backdoor_test_loss = test(args, model, criterion, backdoor_test_loader, epoch,
                                                         type="Backdoor Test")

            # train_clean_loader = DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=True)
            # _, train_clean_loss = test(args, model, criterion, train_clean_loader, epoch, type="Trainset")
            #
            # train_bad_loader = DataLoader(backdoor_train_dataset, batch_size=batch_size, shuffle=True)
            # _, backdoor_loss = test(args, model, criterion, train_bad_loader, epoch, type="Trainset")

            # clean_losses.append(train_clean_loss)
            # bad_losses.append(backdoor_loss)


            scheduler.step()
            if epoch % 5 == 0:
                logger.info("=================Epoch {}=================".format(epoch))
    # np.save("clean_losses_{}".format(args.dataset), np.array(clean_losses))
    # np.save("bad_losses_{}".format(args.dataset), np.array(bad_losses))

    target_dataset = train_dataset
    logger.info(injected_graph_idx)
    losses_bad = []
    losses_clean = []
    losses_all = []
    count_backdoor = []
    model.eval()

    node_matrix = []

    Es = []

    for i in range(int(len(target_dataset))):
        # if i not in injected_graph_idx:
        #     continue
        # logger.info("Graph {}".format(i))
        random_select_nodes = None
        if i in injected_graph_idx:
            count_backdoor.append(i)
            np.random.seed(args.seed)
            if target_dataset[i].x.shape[0] > args.trigger_size:
                random_select_nodes = np.random.choice(target_dataset[i].x.shape[0], args.trigger_size, replace=False)
                logger.info("This is a backdoored graph! " + str(random_select_nodes))
            # else:
            # logger.info(str([j for j in range(target_dataset[i].x.shape[0])]))
        model.embedding_size = args.num_hidden
        model = model.to(device)

        if args.explain_method == "PGExplainer":
            graphs = []
            feats = []
            indices = range(0, len(target_dataset), 5)
            for j in range(0, len(target_dataset), 1):
                graphs.append(target_dataset[j].edge_index)
                feats.append(target_dataset[j].x)
            pgexplainer = PGExplainer.PGExplainer(model.cpu(), graphs, feats, "graph", epochs=50, lr=0.001, )

            pgexplainer.prepare(indices)
            graph, expl_graph_weights = pgexplainer.explain(0)
            print(graph, expl_graph_weights)
        elif args.explain_method == "GNNExplainer":

            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=100, lr=0.02),
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
                explanation_type='model',
                edge_mask_type="object",
                node_mask_type = "object"
            )

            explanation = explainer(target_dataset[i].x.to(device),
                                                                    target_dataset[i].edge_index.to(device))

            # explanation.visualize_graph()
            # plt.show()

            edge_mask = explanation.edge_mask

            node_num = target_dataset[i].x.shape[0]

            if target_dataset[i].edge_index[:, edge_mask >= 0.1].shape[1] != 0:
                masked_edge_index = target_dataset[i].edge_index[:, edge_mask >= 0.1]
                remaining_edge_index = target_dataset[i].edge_index[:, edge_mask < 0.1]
                masked_graph = Data(target_dataset[i].x, remaining_edge_index)
                important_graph = Data(target_dataset[i].x, masked_edge_index)
                original_prediction = torch.argmax(model(target_dataset[i].x.to(device), target_dataset[i].edge_index.to(device)))
                fidelity = model(target_dataset[i].x.to(device), target_dataset[i].edge_index.to(device)).T[original_prediction] - model(masked_graph.x.to(device), masked_graph.edge_index.to(device)).T[original_prediction]
                infidelity = model(target_dataset[i].x.to(device), target_dataset[i].edge_index.to(device)).T[original_prediction] - model(important_graph.x.to(device), important_graph.edge_index.to(device)).T[original_prediction]
                value = fidelity.item() - infidelity.item()
                logger.info("Graph {} ES Value {}".format(i, value))
                Es.append(value)
        elif args.explain_method == "subgraphx":
            # g = torch_geometric.utils.to_networkx(target_dataset[i])
            # nx.draw(g)
            # plt.show()
            explainer = SubgraphX(model, num_classes=args.num_classes, device=device, explain_graph=True, rollout=20)
            _, explanation_results, related_preds = explainer(target_dataset[i].x.to(device),
                                                              target_dataset[i].edge_index.to(device),
                                                              max_nodes=args.trigger_size)
            prediction = model(target_dataset[i].x.to(device), target_dataset[i].edge_index.to(device), ).argmax(
                -1).item()
            explanation_results = explainer.read_from_MCTSInfo_list(explanation_results)[prediction]
            explainer.visualization(explanation_results, args.trigger_size, PlotUtils(args.dataset))
            plt.show()
            nodelist = find_closest_node_result(explanation_results, max_nodes=args.trigger_size + 2).coalition
            # logger.info(nodelist)

            ###################
            remaining = [i for i in range(target_dataset[i].x.shape[0]) if i not in nodelist]
            random.shuffle(remaining)
            ret = nodelist + remaining
            node_matrix.append(ret)
            ################
            edgelist = torch.LongTensor([(n_frm.item(), n_to.item()) for (n_frm, n_to) in target_dataset[i].edge_index.T
                                         if n_frm.item() in nodelist or n_to.item() in nodelist])
            target_dataset[i].edge_index = edgelist.T
            if edgelist.shape[0] != 0:
                edge_index, edge_attr, mask = torch_geometric.utils.remove_isolated_nodes(target_dataset[i].edge_index,
                                                                                          num_nodes=
                                                                                          target_dataset[i].x.shape[0])
                # print(mask, edge_index)
                target_dataset[i].x = target_dataset[i].x[mask, :]
                target_dataset[i].edge_index = edge_index
                target_dataset[i].num_nodes = target_dataset[i].x.shape[0]
                target_dataset[i].edge_attr = None
                prepare_dataset_x([target_dataset[i]], num_features - 1)

        elif args.explain_method == "degree":
            hier_explain(train_dataset, model, i, class_idx=target_dataset[i].y.item())
        else:
            raise NotImplementedError

    y_label = np.zeros((len(target_dataset), 1))
    y_label[count_backdoor, :] = 1

    tps = []
    fps = []
    for threshold in reversed([0, 1E-10, 1E-9, 1E-8, 1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.1, 0.2, 0.5, 0.8,  1, 2, 3, max(Es) + 1]):
        sort_index = np.argwhere(np.array(Es) > threshold).T[0].tolist()
        # cross = set(sort_index) & set(count_backdoor)
        #
        # precision = len(cross) / len(sort_index) # TP
        # recall = len(cross) / len(count_backdoor)
        y_pred = np.zeros((len(target_dataset), 1))
        y_pred[sort_index] = 1
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
        tps.append(tp / len(count_backdoor))
        fps.append(fp / (len(target_dataset) - len(count_backdoor)))

        accuracy = accuracy_score(y_label, y_pred)

        logger.info("[{}] Accuracy Score {}".format(threshold, accuracy))

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
