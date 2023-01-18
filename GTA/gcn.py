import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(''))
import dgl
import dgl.function as fn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from utils.graph import numpy_to_graph

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
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


# 2 layers by default
class GCN_G(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
                 dropout=0.2,
                 activation=F.relu):
        super(GCN_G, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.layers.append(GCNLayer(hidden_dim[i], hidden_dim[i+1]))

        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        fc.append(nn.Linear(hidden_dim[-1], out_dim))
        self.fc = nn.Sequential(*fc)

    # def __init__(self, in_dim, out_dim,
    #              hidden_dim=[64, 32],  # GNN layers + 1 layer MLP
    #              dropout=0.2,
    #              activation=F.relu):
    #     super(GCN_G, self).__init__()
    #
    #     self.layers = nn.ModuleList()
    #     self.layers.append(GCNConv(in_dim, hidden_dim[0]))
    #     for i in range(len(hidden_dim) - 1):
    #         self.layers.append(GCNConv(hidden_dim[i], hidden_dim[i+1]))
    #         self.layers.append(torch.nn.BatchNorm1d(hidden_dim[i+1]))
    #
    #     self.fc1 = Linear(hidden_dim[-1], hidden_dim[-1])
    #     self.fc2 = Linear(hidden_dim[-1], out_dim)
    def forward(self, data):
        batch_g = []
        for adj in data[1]:
            # print(adj.shape)
            batch_g.append(numpy_to_graph(adj.cpu().detach().T.numpy(), to_cuda=adj.is_cuda))
        batch_g = dgl.batch(batch_g)

        mask = data[2]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2) # (B,N,1)

        B,N,F = data[0].shape[:3]
        x = data[0].reshape(B*N, F)
        mask = mask.reshape(B*N, 1)
        # print(x.shape)
        for layer in self.layers:
            x = layer(batch_g, x)
            x = x * mask


        F_prime = x.shape[-1]
        x = x.reshape(B, N, F_prime)
        # print(x.shape)
        # input()
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        # x = torch.mean(x, dim=1).squeeze()
        x = self.fc(x)
        # x = self.fc2(x)
        return x

# class GCN(torch.nn.Module):
#     def __init__(self, num_features=1, num_classes=1, num_hidden=32):
#         super(GCN, self).__init__()
#
#         # if data.x is None:
#         #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
#         # dataset.data.edge_attr = None
#
#         # num_features = dataset.num_features
#         dim = num_hidden
#
#         self.conv1 = GCNConv(num_features, dim)
#         self.bn1 = torch.nn.BatchNorm1d(dim)
#
#         self.conv2 = GCNConv(dim,dim)
#         self.bn2 = torch.nn.BatchNorm1d(dim)
#
#         self.conv3 = GCNConv(dim,dim)
#         self.bn3 = torch.nn.BatchNorm1d(dim)
#
#         self.conv4 = GCNConv(dim,dim)
#         self.bn4 = torch.nn.BatchNorm1d(dim)
#
#         self.conv5 = GCNConv(dim,dim)
#         self.bn5 = torch.nn.BatchNorm1d(dim)
#
#         self.fc1 = Linear(dim, dim)
#         self.fc2 = Linear(dim, num_classes)
#
#     def forward(self, x, edge_index, batch=None, edge_weights=None):
#         x = self.embedding(x, edge_index, edge_weights)
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.fc1(x))
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=-1)
#
#     def embedding(self, x, edge_index, edge_weights=None):
#         if edge_weights is None:
#              edge_weights = torch.ones(edge_index.size(1)).to(x.device)
#         out1 = self.conv1(x, edge_index, edge_weights)
#         out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
#         out1 = F.relu(out1)
#
#         out2 = self.conv2(out1, edge_index, edge_weights)
#         out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
#         out2 = F.relu(out2)
#
#         out3 = self.conv3(out2, edge_index, edge_weights)
#         out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
#         out3 = F.relu(out3)
#
#         out4 = self.conv4(out3, edge_index, edge_weights)
#         out4 = torch.nn.functional.normalize(out4, p=2, dim=1)  # this is not used in PGExplainer
#         out4 = F.relu(out4)
#
#         out5 = self.conv5(out4, edge_index, edge_weights)
#         out5 = torch.nn.functional.normalize(out5, p=2, dim=1)  # this is not used in PGExplainer
#         out5 = F.relu(out5)
#
#
#         input_lin = out5
#
#         return input_lin
    
