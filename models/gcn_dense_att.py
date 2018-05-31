import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj_set, att):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        support = torch.mm(inputs, self.w) + self.b
        outputs = None
        for i, adj in enumerate(adj_set):
            y = torch.mm(adj, support) * att[i]
            if outputs is None:
                outputs = y
            else:
                outputs = outputs + y

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


class GCN_Dense_Att(nn.Module):

    def __init__(self, n, edges_set, in_channels, out_channels, hidden_layers):
        super().__init__()

        self.n = n
        self.d = len(edges_set)

        self.a_adj_set = []
        self.r_adj_set = []

        for edges in edges_set:
            edges = np.array(edges)
            adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                shape=(n, n), dtype='float32')
            a_adj = spm_to_tensor(normt_spm(adj, method='in')).cuda()
            r_adj = spm_to_tensor(normt_spm(adj.transpose(), method='in')).cuda()
            self.a_adj_set.append(a_adj)
            self.r_adj_set.append(r_adj)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        self.a_att = nn.Parameter(torch.ones(self.d))
        self.r_att = nn.Parameter(torch.ones(self.d))

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        graph_side = True
        for conv in self.layers:
            if graph_side:
                adj_set = self.a_adj_set
                att = self.a_att
            else:
                adj_set = self.r_adj_set
                att = self.r_att
            att = F.softmax(att, dim=0)
            x = conv(x, adj_set, att)
            graph_side = not graph_side

        return F.normalize(x)

