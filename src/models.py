import torch_geometric
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge, BatchNorm
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from math import ceil
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge, global_add_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, global_mean_pool, JumpingKnowledge
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from math import ceil

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

        self.embedding_size = 128*5

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if edge_weights is None:
             edge_weights = torch.ones((edge_index.shape[1]))
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        # x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
             edge_weights = torch.ones(edge_index.size(1))
        out1 = self.conv1(x, edge_index, None)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index, None)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index, None)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = F.relu(out3)

        input_lin = out3

        return input_lin



class GCN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GCN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        self.conv1 = GCNConv(num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = GCNConv(dim,dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = GCNConv(dim,dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = GCNConv(dim,dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = GCNConv(dim,dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        x = self.embedding(x, edge_index, edge_weights)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
             edge_weights = torch.ones(edge_index.size(1)).to(x.device)
        # else:
        #     print(edge_weights)
        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = F.relu(out3)

        out4 = self.conv4(out3, edge_index, edge_weights)
        out4 = torch.nn.functional.normalize(out4, p=2, dim=1)  # this is not used in PGExplainer
        out4 = F.relu(out4)

        out5 = self.conv5(out4, edge_index, edge_weights)
        out5 = torch.nn.functional.normalize(out5, p=2, dim=1)  # this is not used in PGExplainer
        out5 = F.relu(out5)


        input_lin = out3

        return input_lin


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GraphSAGE, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        self.conv1 = SAGEConv(num_features, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.conv2 = SAGEConv(dim,dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.conv3 = SAGEConv(dim,dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.conv4 = SAGEConv(dim,dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.conv5 = SAGEConv(dim,dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.conv6 = SAGEConv(dim, dim)
        self.bn6 = torch.nn.BatchNorm1d(dim)

        self.conv7 = SAGEConv(dim, dim)
        self.bn7 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = F.relu(self.conv7(x, edge_index))
        x = self.bn7(x)
        x = gap(x, batch)
        # x = TopKPooling(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
class NodeGCN(torch.nn.Module):
    """
    A graph clasification models for nodes decribed in https://arxiv.org/abs/2011.04573.
    This models consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super(NodeGCN, self).__init__()
        self.embedding_size = 20 * 3
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(20)        # BN is not used in GNNExplainer
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(20)
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin


import dgl
import dgl.function as fn
from GTA.utils.graph import numpy_to_graph
import sys
sys.path.append("/home/zg37632/g-mixup/")
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


# Used for inductive case (graph classification) by default.
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g = g.cpu()
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


# 2 layers by default
class GCN_N(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu):
        super(GCN_N, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i + 1]))

        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, edge_index):
        batch_g = []
        adj = torch_geometric.utils.to_dense_adj(edge_index)
        batch_g.append(numpy_to_graph(adj[0].cpu().T.numpy()))
        if adj.shape[1] != x.shape[0]:
            return 123
        # for adj in data[1]:
        #     batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy(), to_cuda=adj.is_cuda))
        batch_g = dgl.batch(batch_g)

        x = torch.Tensor(x)

        for layer in self.layers:
            x = layer(batch_g, x)
            # x = x * mask
        x = torch.max(x, dim=0)[0].squeeze()  # max pooling over nodes (usually performs better than average)

        x = self.fc(x)
        return x
