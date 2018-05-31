import os
import os.path as osp
import shutil

import numpy as np
import scipy.sparse as sp
import torch


def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu {}'.format(gpu))


def pick_vectors(dic, wnids, is_tensor=False):
    o = next(iter(dic.values()))
    dim = len(o)
    ret = []
    for wnid in wnids:
        v = dic.get(wnid)
        if v is None:
            if not is_tensor:
                v = [0] * dim
            else:
                v = torch.zeros(dim)
        ret.append(v)
    if not is_tensor:
        return torch.FloatTensor(ret)
    else:
        return torch.stack(ret)


def l2_loss(a, b):
    return ((a - b)**2).sum() / (len(a) * 2)


def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack(
            (sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

