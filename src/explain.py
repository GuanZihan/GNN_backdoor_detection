from utils.utils import *
import torch
from CD_layers import *
import time
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def find_edge(model, dataset, idx, class_idx=None, node_sort=None, topk=None, start_num=2):
    if node_sort is None:
        node_sort, node_color = print_explain(dataset, model, idx, class_idx=class_idx, visible=False)
    if class_idx is None:
        class_idx = int(dataset[idx].y[0])
    data = get_data(dataset, idx)
    data = data.to(device)
    start_num = start_num

    rest = list(node_sort.copy())
    select = set()
    neighbor_nodes = set()
    select_node_progress = []
    edge_index = np.array(dataset[idx].edge_index)

    while len(select) < topk and len(rest) != 0:
        if len(neighbor_nodes) == 0:
            for _ in range(start_num):
                if len(rest) == 0:
                    pass
                select_node = rest[0]
                select.add(select_node)
                rest.remove(select_node)
                # add new neighbor edge
                temp_neighbor_idx = (np.where(edge_index[0] == select_node))[0]
                # print(temp_neighbor_idx)
                for n_idx in temp_neighbor_idx:
                    n = int(edge_index[1][n_idx])
                    if n not in select:
                        neighbor_nodes.add(n)
                try:
                    neighbor_nodes.remove(select_node)
                except:
                    pass
                # print('no neighbor')
                # print('select: ', select)
                # print('neighbor_nodes', neighbor_nodes)
            select_node_progress.append(select.copy())

        else:
            # prepare list for score
            check_list = []
            for node in neighbor_nodes:
                group = select.copy()
                group.add((int)(node))
                check_list.append(group)
            # print(check_list)
            class_score = get_score(model, data, input_mask_list=check_list.copy())
            group_scores = class_score[class_idx]['rel']
            group_rank = np.argsort(group_scores)[::-1]
            # print(group_rank)
            select = check_list[group_rank[0]]
            # print(check_list)
            intersect = select.intersection(neighbor_nodes)
            # print(intersect)
            neighbor_nodes = neighbor_nodes - select
            # print(rest)
            for node in intersect:
                rest.remove(node)
                # add new neighbor edge
                temp_neighbor_idx = (np.where(edge_index[0] == node))[0]
                # print('temp_neighbor_idx', temp_neighbor_idx)
                for n_idx in temp_neighbor_idx:
                    n = int(edge_index[1][n_idx])
                    if n not in select:
                        neighbor_nodes.add(n)
            select_node_progress.append(select.copy())
        size = max(edge_index[0]) + 1
        adj = np.zeros((size, size))
        edge_pred = []
        for i in range(edge_index.shape[1]):
            r = edge_index[0][i]
            c = edge_index[1][i]
            if r in select and c in select:
                adj[r][c] = 1
                adj[c][r] = 1

            if r < c:
                if r in select and c in select:
                    edge_pred.append(1)
                else:
                    edge_pred.append(0)

        adj = preprocess_adj(adj)
    return select, select_node_progress, adj, edge_pred


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def CD_explain(model, dataset, idx=0, mask_node_list=None, node_range=None, target_node=None):
    """
    idx: idx of graph to explain
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data = get_data(dataset, idx)
    data = data.to(device)

    mask_list = []
    # generate mask
    node_range = range(data.x.shape[0]) if node_range is None else node_range
    target_node = target_node if target_node is not None else node_range[0]
    '''
    for i in range(node_range):
        # generate data idx mask to check
        mask_index = [0] * data.x.shape[0]
        mask_index[i] = 1
     b mask_node_list is not None and i in mask_node_list:
            for n in mask_node_list:
                print(i)
                print(mask_node_list)
                mask_index[n] = 1
        mask_list.append(mask_index)
    '''

    for i in node_range:
        if mask_node_list is not None and i in mask_node_list:
            mask_list.append(mask_node_list)
        else:
            mask_list.append([i])
    # forward to explain according to mask list
    print('target node: ', target_node)
    class_score = get_score(model=model, data=data, input_mask_list=mask_list, target_node=target_node)
    return class_score


def get_score(model, data, input_mask_list, softmax=True, target_node=0):
    Binary_mask_list = []
    for i in range(len(input_mask_list)):
        mask_index = [0] * data.x.shape[0]
        for m in input_mask_list[i]:
            mask_index[m] = 1
        Binary_mask_list.append(mask_index)

    preds_list = []
    model.eval()
    for mask_index in Binary_mask_list:
        model.zero_grad()
        preds = model.forward(data, CD_explain=True, mask_index=mask_index)
        # print(preds)
        # print(data.y)
        preds['rel'] = preds['rel'].T
        preds['irrel'] = preds['irrel'].T
        # print(preds['irrel'].shape[0])
        # print(dataset.num_classes)
        if softmax:
            preds = CD_softmax(preds, index=torch.tensor([0] * preds['irrel'].shape[0]).to(device))
        # print(preds)
        preds_list.append(preds)
        # need to softmax??
        # print('preds shape', preds['rel'].shape)
    # rel cd score for each class
    class_score = {}
    for class_idx in range(preds['irrel'].shape[0]):
        class_score[class_idx] = {}
        class_score[class_idx]['rel'] = []
        class_score[class_idx]['irrel'] = []
        for preds in preds_list:
            class_score[class_idx]['rel'].append((float)((preds['rel'][class_idx][target_node].cpu().detach())))
            class_score[class_idx]['irrel'].append((float)((preds['irrel'][class_idx][target_node].cpu().detach())))
            # class_score[class_idx]['rel'].append((float)((preds['rel'][0][class_idx].cpu().detach())))
            # class_score[class_idx]['irrel'].append((float)((preds['irrel'][0][class_idx].cpu().detach())))

    return class_score


def print_explain(dataset, model, idx, class_idx=None, visible=True, figsize=None, node_range=None, **kwargs):
    class_idx = class_idx if class_idx is not None else (int)(dataset[idx].y[0])

    begin_time = time.time()
    class_score = CD_explain(model=model, dataset=dataset, idx=idx, node_range=node_range)
    # print(class_score)
    elapsed = time.time() - begin_time
    node_colors = class_score[class_idx]['rel']

    print('epoch time: ', elapsed)

    if visible:
        dataset.subgraph = True
        if figsize is not None:
            plt.figure(figsize=figsize)
        G = to_networkx(dataset[idx])

        pos = nx.kamada_kawai_layout(G)
        size_ratio = 1.0 + (node_colors - np.min(node_colors)) / (np.max(node_colors) - np.min(node_colors))
        node_sizes = 800 * size_ratio
        node_colors = class_score[class_idx]['rel']
        cmap = plt.cm.Oranges

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            edgecolors='blue'
        )
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="-",
            arrowsize=10,
            width=2,
        )
        label_list = {}
        for i in range(dataset[idx].x.shape[0]):
            label_list[i] = str(i)
            if 'node_name_list' in kwargs:
                node_name_idx = int(np.argmax(dataset[idx].x[i]))
                label_list[i] += ':' + kwargs['node_name_list'][node_name_idx]

        labels = nx.draw_networkx_labels(G, pos, label_list, font_size=12, font_color="black")
        plt.colorbar(nodes)

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()

    return np.argsort(node_colors)[::-1], node_colors, elapsed


def Edge_explain(model, dataset, edge_list=None, idx=0, node_range=None, softmax=False):
    """
    idx: idx of graph to explain
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data = get_data(dataset, idx)
    data = data.to(device)

    mask_list = []
    # generate mask
    node_range = data.x.shape[0] if node_range is None else node_range

    if edge_list is not None:
        mask_list = edge_list
        print('have edge_list')
    else:
        mask_list = []
        for _ in range(len(data.edge_index[0])):
            mask_list.append([int(data.edge_index[0][_]), int(data.edge_index[1][_])])
    # for i in range(len(mask_list)):
    #    print(i, ' : ', mask_list[i])

    # forward to explain according to mask list
    begin_time = time.time()
    class_score = get_score(model, data, mask_list, softmax=softmax)
    elapsed = time.time() - begin_time
    return class_score, elapsed


def print_subgraph_explain(dataset, model, idx=0, class_idx=None, visible=True, figsize=None, node_range=None,
                           **kwargs):
    class_idx = class_idx if class_idx is not None else (int)(dataset[idx].y[0])

    begin_time = time.time()
    class_score = CD_explain(model=model, dataset=dataset, idx=idx, node_range=node_range)
    # print(class_score)
    elapsed = time.time() - begin_time
    node_colors = class_score[class_idx]['rel']

    print('epoch time: ', elapsed)

    if visible:

        if figsize is not None:
            plt.figure(figsize=figsize)
        idx = idx if node_range is None else node_range[0]

        if_sub = dataset.subgraph
        dataset.subgraph = True
        data = dataset[idx]
        dataset.subgraph = if_sub

        G = to_networkx(data)

        pos = nx.kamada_kawai_layout(G)
        size_ratio = 1.0 + (node_colors - np.min(node_colors)) / (np.max(node_colors) - np.min(node_colors))
        node_sizes = 800 * size_ratio
        node_colors = class_score[class_idx]['rel']
        cmap = plt.cm.Oranges

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            edgecolors='blue'
        )
        edges = nx.draw_networkx_edges(
            G,
            pos,
            node_size=node_sizes,
            arrowstyle="-",
            arrowsize=10,
            width=2,
        )
        label_list = {}
        for i in range(data.x.shape[0]):
            label_list[i] = str(i)
            if 'node_name_list' in kwargs:
                node_name_idx = int(np.argmax(data.x[i]))
                label_list[i] += ':' + kwargs['node_name_list'][node_name_idx]

        labels = nx.draw_networkx_labels(G, pos, label_list, font_size=12, font_color="black")
        plt.colorbar(nodes)

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()

    return np.argsort(node_colors)[::-1], node_colors, elapsed