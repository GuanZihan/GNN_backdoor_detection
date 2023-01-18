from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from torch_geometric.nn import GCNConv, GATConv
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros, kaiming_uniform
from torch.nn import Linear
from typing import Optional
from torch_scatter import scatter
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn import init
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn.modules.batchnorm import _NormBase


class CD_BatchNorm(torch.nn.Module):

    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(CD_BatchNorm, self).__init__()
        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)

    def reset_parameters(self):
        self.module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""

        if isinstance(x, dict):
            result = {}
            # t = x['rel'] + x['irrel']
            # device = 'cuda' if t.is_cuda else 'cpu'
            # mean = torch.mean(t, 0)
            # var = torch.var(t, 0)
            # self.module.running_mean = torch.nn.parameter.Parameter(data = mean, requires_grad = False)
            # self.module.running_var = torch.nn.parameter.Parameter(data = var, requires_grad = False)
            # self.module.to(device)
            result['rel'] = self.module(x['rel'])
            result['irrel'] = self.module(x['irrel'])
            # print(t - x['rel'] - x['irrel'])

            # self.module.running_mean = None
            # self.module.running_var = None
            return result
        else:

            return self.module(x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'


class CD_GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(CD_GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if isinstance(x, dict):
            node_num = x['rel'].size(self.node_dim)
            dtype = x['rel'].dtype
        else:
            node_num = x.size(self.node_dim)
            dtype = x.dtype

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, node_num,
                        self.improved, self.add_self_loops, dtype=dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, node_num,
                        self.improved, self.add_self_loops, dtype=dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if isinstance(x, dict):
            rel = x['rel']
            irrel = x['irrel']

            # print('rel shape: ', rel.shape)
            # print('weight shape: ', self.weight.shape)
            rel = torch.matmul(rel, self.weight)
            irrel = torch.matmul(irrel, self.weight)
            # print('seperate matmul:')
            # print(rel+irrel)

            rel = self.propagate(edge_index, x=rel, edge_weight=edge_weight,
                                 size=None)
            irrel = self.propagate(edge_index, x=irrel, edge_weight=edge_weight,
                                   size=None)

            # print('seperate propagate:')
            # print(rel+irrel)
            if self.bias is not None:
                abs_rel = torch.abs(rel)
                abs_irrel = torch.abs(irrel)
                bias_rel = self.bias * (abs_rel / (abs_rel + abs_irrel))
                bias_irrel = self.bias - bias_rel
                rel += bias_rel
                irrel += bias_irrel
                # irrel += self.bias
                # rel += self.bias

            # print('seperate propagate:')
            # print(rel+irrel)
            out = {}
            out['rel'] = rel
            out['irrel'] = irrel
        else:
            # print(self.weight.shape)
            x = torch.matmul(x, self.weight)
            # print('union matmul:')
            # print(x)
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                 size=None)
            # print('union propagate:')
            # print(out)
            if self.bias is not None:
                out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return x_j
        else:
            return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def CD_global_mean_pool(x, batch, size: Optional[int] = None):
    size = int(batch.max().item() + 1) if size is None else size
    if (isinstance(x, dict)):
        rel = x['rel']
        irrel = x['irrel']
        x['rel'] = scatter(rel, batch, dim=0, dim_size=size, reduce='mean')
        x['irrel'] = scatter(irrel, batch, dim=0, dim_size=size, reduce='mean')
        return x
    else:

        return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


def CD_global_sum_pool(x, batch, size: Optional[int] = None):
    size = int(batch.max().item() + 1) if size is None else size
    if (isinstance(x, dict)):
        rel = x['rel']
        irrel = x['irrel']
        x['rel'] = scatter(rel, batch, dim=0, dim_size=size, reduce='sum')
        x['irrel'] = scatter(irrel, batch, dim=0, dim_size=size, reduce='sum')
        return x
    else:

        return scatter(x, batch, dim=0, dim_size=size, reduce='sum')


def CD_feature_max_pool(inputs, batch, batch_size=25):
    if isinstance(inputs, dict):
        _, indices = torch.max(inputs['rel'] + inputs['irrel'], dim=-1, keepdim=False)
        rel = torch.zeros_like(_)
        irrel = torch.zeros_like(_)
        for i in range(indices.shape[0]):
            rel[i] = inputs['rel'][i][indices[i]]
            irrel[i] = inputs['irrel'][i][indices[i]]
        rel = rel.reshape(-1, batch_size)
        irrel = irrel.reshape(-1, batch_size)
        output = {'rel': rel, 'irrel': irrel}
        return output

    output, indices = torch.max(inputs, dim=-1, keepdim=True)
    # print(output.shape)
    output = output.reshape(-1, batch_size)
    return output


def CD_feature_mean_pool(inputs, batch, batch_size=25):
    if isinstance(inputs, dict):
        rel = torch.mean(inputs['rel'], dim=-1, keepdim=True).reshape(-1, batch_size)
        irrel = torch.mean(inputs['irrel'], dim=-1, keepdim=True).reshape(-1, batch_size)
        output = {'rel': rel, 'irrel': irrel}
        return output

    output = torch.mean(inputs, dim=-1, keepdim=True)
    # print(output.shape)
    output = output.reshape(-1, batch_size)
    return output


def CD_global_max_pool(inputs, batch):
    size = int(batch.max().item() + 1)
    cd_explain = isinstance(inputs, dict)

    if not cd_explain:
        output = scatter(inputs, batch, dim=0, dim_size=size, reduce='max')

        return output

    x = inputs['rel'] + inputs['irrel']
    rel_max_list = []
    irrel_max_list = []
    for r in range(size):
        value, indices = torch.max(x[batch == r], dim=0)
        rel_tmp = torch.zeros_like(inputs['rel'][0])
        d = rel_tmp.shape[-1]
        for i in range(d):
            rel_tmp[i] = inputs['rel'][batch == r][indices[i]][i]
        rel_max_list.append(rel_tmp)

        irrel_tmp = torch.zeros_like(inputs['irrel'][0])
        d = irrel_tmp.shape[-1]
        for i in range(d):
            irrel_tmp[i] = inputs['irrel'][batch == r][indices[i]][i]
        irrel_max_list.append(irrel_tmp)

    result = {}
    result['rel'] = torch.stack(rel_max_list, dim=0)
    result['irrel'] = torch.stack(irrel_max_list, dim=0)

    return result


def CD_relu(inputs: Tensor, inplace: bool = False) -> Tensor:
    if isinstance(inputs, dict):
        rel = inputs['rel']
        irrel = inputs['irrel']
        result = {}
        # result['rel'] = (F.relu(rel) + F.relu(rel+irrel) - F.relu(irrel)) * 0.5
        # result['irrel'] = (F.relu(rel + irrel) - F.relu(rel) + F.relu(irrel)) * 0.5
        result['rel'] = F.relu(rel)
        result['irrel'] = F.relu(rel + irrel) - F.relu(rel)

    else:
        result = F.relu(inputs, inplace)

    return result


def CD_leaky_relu(inputs: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if isinstance(inputs, dict):
        rel = inputs['rel']
        irrel = inputs['irrel']
        result = {}
        result['rel'] = F.leaky_relu(rel, negative_slope, inplace)
        result['irrel'] = F.leaky_relu(rel + irrel, negative_slope, inplace) - F.leaky_relu(rel, negative_slope,
                                                                                            inplace)

    else:
        result = F.leaky_relu(inputs, negative_slope, inplace)

    return result


class CD_Linear(Linear):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(CD_Linear, self).__init__(in_features, out_features, bias)
        # self.bias = bias
        # print('if bias: ', bias)

    def forward(self, inputs: Tensor) -> Tensor:

        if isinstance(inputs, dict):
            # print('in cd linear dict')
            result = {}
            # need to seperate bias ?? bias in irrel ??
            # result['rel'] = F.linear(inputs['rel'], self.weight)
            # result['irrel'] = F.linear(inputs['irrel'], self.weight, self.bias)

            # result['rel'] = F.linear(inputs['rel'], self.weight, self.bias)
            # result['irrel'] = F.linear(inputs['irrel'], self.weight)

            result['rel'] = F.linear(inputs['rel'], self.weight)
            # result['irrel'] = F.linear(inputs['irrel'], self.weight)
            result['irrel'] = F.linear(inputs['irrel'] + inputs['rel'], self.weight) - result['rel']
            if self.bias is not None:
                rel_prop = torch.abs(result['rel']) / (torch.abs(result['rel']) + torch.abs(result['irrel']))
                irrel_prop = 1 - rel_prop
                # print('has bias')
                # print('rel_prop: ', rel_prop)
                bias_rel = self.bias * rel_prop
                bias_irrel = self.bias - bias_rel
                result['rel'] += bias_rel
                result['irrel'] += bias_irrel
        else:
            # print('in cd linear')
            result = F.linear(inputs, self.weight, self.bias)
        return result


from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr

from torch_geometric.utils.num_nodes import maybe_num_nodes


def CD_softmax(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
    cd_explain = isinstance(src, dict)
    if cd_explain:
        out = src['rel'] + src['irrel']
        out_rel = src['rel']
        out_irrel = src['irrel']
    else:
        out = src

    # print(out.shape)

    out = out - out.max()
    out = out.exp()
    if cd_explain:
        _rel = 1.0 / (1.0 + (torch.abs(out_irrel) - torch.abs(out_rel)).exp())
        _irrel = 1.0 - _rel

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
    elif index is None:
        index = torch.tensor([0] * out.shape[0])

    else:
        raise NotImplementedError

    if cd_explain:
        return {'rel': (out / (out_sum + 1e-16)) * _rel,
                'irrel': (out / (out_sum + 1e-16)) - (out / (out_sum + 1e-16)) * _rel}
    else:
        return out / (out_sum + 1e-16)


class CD_GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(CD_GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = CD_Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = CD_Linear(in_channels[0], heads * out_channels, bias=False)
            self.lin_r = CD_Linear(in_channels[1], heads * out_channels, bias=False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self._rel = None
        self._irrel = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, ):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        cd_explain = isinstance(x, dict)
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            # print('be here 1')
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            # print(x.shape)
            self.x = x
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            self.x_l = x_l
            self.x_r = x_r
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
            self.alpha_l = alpha_l
            self.alpha_r = alpha_r
        elif cd_explain:
            self.x_dict = x
            x_rel = x['rel']
            x_irrel = x['irrel']

            x_l = self.lin_l(x)
            x_rel_l = x_l['rel'].view(-1, H, C)
            x_irrel_l = x_l['irrel'].view(-1, H, C)

            x_rel_r = x_rel_l
            x_irrel_r = x_irrel_l
            self.x_rel_l = x_rel_l
            self.x_rel_r = x_rel_r
            self.x_irrel_l = x_irrel_l
            self.x_irrel_r = x_irrel_r

            alpha_rel_l = (x_rel_l * self.att_l).sum(dim=-1)
            alpha_rel_r = (x_rel_r * self.att_r).sum(dim=-1)

            alpha_irrel_l = (x_irrel_l * self.att_l).sum(dim=-1)
            alpha_irrel_r = (x_irrel_r * self.att_r).sum(dim=-1)

            x_l = x_rel_l + x_irrel_l
            alpha_l = alpha_rel_l + alpha_irrel_l
            alpha_r = alpha_rel_r + alpha_irrel_r
            self.alpha_l_prime = alpha_l
            self.alpha_r_prime = alpha_r
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)

        x_rel = None
        x_irrel = None
        alpha_rel = None
        alpha_irrel = None
        if cd_explain:
            x_rel = (x_rel_l, x_rel_r)
            x_irrel = (x_irrel_l, x_irrel_r)
            alpha_rel = (alpha_rel_l, alpha_rel_r)
            alpha_irrel = (alpha_irrel_l, alpha_irrel_r)

            # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        if not cd_explain:

            out = self.propagate(edge_index, x=(x_l, x_r),
                                 alpha=(alpha_l, alpha_r), size=size,
                                 cd_explain=cd_explain,
                                 is_rel=False,
                                 x_rel=x_rel,
                                 x_irrel=x_irrel,
                                 alpha_rel=alpha_rel,
                                 alpha_irrel=alpha_irrel)
        else:
            out = {}

            out['rel'] = self.propagate(edge_index, x=(x_l, x_r),
                                        alpha=(alpha_l, alpha_r), size=size,
                                        cd_explain=cd_explain,
                                        is_rel=True,
                                        x_rel=x_rel,
                                        x_irrel=x_irrel,
                                        alpha_rel=alpha_rel,
                                        alpha_irrel=alpha_irrel)
            out['irrel'] = self.propagate(edge_index, x=(x_l, x_r),
                                          alpha=(alpha_l, alpha_r), size=size,
                                          cd_explain=cd_explain,
                                          is_rel=False,
                                          x_rel=x_rel,
                                          x_irrel=x_irrel,
                                          alpha_rel=alpha_rel,
                                          alpha_irrel=alpha_irrel)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            if not cd_explain:
                out = out.view(-1, self.heads * self.out_channels)
            else:
                out['rel'] = out['rel'].view(-1, self.heads * self.out_channels)
                out['irrel'] = out['irrel'].view(-1, self.heads * self.out_channels)
        else:
            if not cd_explain:
                out = out.mean(dim=1)
            else:
                out['rel'] = out['rel'].mean(dim=1)
                out['irrel'] = out['irrel'].mean(dim=1)

        if self.bias is not None:
            if not cd_explain:
                out += self.bias
            else:
                # out['irrel'] += self.bias

                abs_rel = torch.abs(out['rel'])
                abs_irrel = torch.abs(out['irrel'])
                bias_rel = self.bias * (abs_rel / (abs_rel + abs_irrel))
                bias_irrel = self.bias - bias_rel
                out['rel'] += bias_rel
                out['irrel'] += bias_irrel

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                cd_explain: bool,
                is_rel: bool,
                x_rel_j: Tensor, x_irrel_j: Tensor,
                alpha_rel_j: Tensor, alpha_rel_i: Tensor,
                alpha_irrel_j: Tensor, alpha_irrel_i: Tensor,
                size_i: Optional[int]) -> Tensor:
        if not cd_explain:
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
            alpha = F.leaky_relu(alpha, self.negative_slope)

            alpha = CD_softmax(alpha, index, ptr, size_i)
            self.alpha = alpha

            self._alpha = alpha
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            self._x_j = x_j * alpha.unsqueeze(-1)
            return x_j * alpha.unsqueeze(-1)
        else:
            # if not is_rel:
            #    return self._irrel
            alpha_rel = alpha_rel_j if alpha_rel_i is None else alpha_rel_j + alpha_rel_i
            alpha_irrel = alpha_irrel_j if alpha_irrel_i is None else alpha_irrel_j + alpha_irrel_i
            # alpha = alpha_rel + alpha_irrel
            alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

            alpha_dict = {}
            alpha_dict['rel'] = alpha_rel
            alpha_dict['irrel'] = alpha_irrel
            alpha_dict = CD_leaky_relu(alpha_dict, self.negative_slope)
            alpha = CD_leaky_relu(alpha, self.negative_slope)

            # print('use cd softmax')
            alpha_dict = CD_softmax(alpha_dict, index, ptr, size_i)
            alpha = CD_softmax(alpha, index, ptr, size_i)
            self.alpha_dict = alpha_dict
            self.alpha_prime = alpha

            self._alpha = alpha
            result = {}
            result['rel'] = x_rel_j * alpha_dict['rel'].unsqueeze(-1)
            result['irrel'] = x_irrel_j * alpha_dict['rel'].unsqueeze(-1) + x_irrel_j * alpha_dict['irrel'].unsqueeze(
                -1) + x_rel_j * alpha_dict['irrel'].unsqueeze(-1)
            # result['rel'] = x_rel_j * alpha.unsqueeze(-1)
            # result['irrel'] = x_irrel_j * alpha.unsqueeze(-1)
            # self._rel = result['rel']
            # self._irrel = result['irrel']
            return result['rel'] if is_rel else result['irrel']

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)