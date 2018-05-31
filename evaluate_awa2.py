import argparse
import json
import os.path as osp

import torch
from torch.utils.data import DataLoader

from models.resnet import make_resnet50_base
from datasets.image_folder import ImageFolder
from utils import set_gpu, pick_vectors


def test_on_subset(dataset, cnn, n, pred_vectors, all_label,
                   consider_trains):
    hit = 0
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch 
        data = data.cuda()

        feat = cnn(data) # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1)

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        pred = torch.argmax(table, dim=1)
        hit += (pred == all_label).sum().item()
        tot += len(data)

    return hit, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn')
    parser.add_argument('--pred')

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    set_gpu(args.gpu)

    awa2_split = json.load(open('materials/awa2-split.json', 'r'))
    train_wnids = awa2_split['train']
    test_wnids = awa2_split['test']

    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True).cuda()
    pred_vectors = pred_vectors.cuda()

    n = len(train_wnids)
    m = len(test_wnids)
    
    cnn = make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn = cnn.cuda()
    cnn.eval()

    test_names = awa2_split['test_names']

    ave_acc = 0; ave_acc_n = 0

    results = {}

    awa2_path = 'materials/datasets/awa2'

    for i, name in enumerate(test_names, 1):
        dataset = ImageFolder(osp.join(awa2_path, 'JPEGImages'), [name], 'test')
        hit, tot = test_on_subset(dataset, cnn, n, pred_vectors, n + i - 1,
                                  args.consider_trains)
        acc = hit / tot
        ave_acc += acc
        ave_acc_n += 1

        print('{} {}: {:.2f}%'.format(i, name.replace('+', ' '), acc * 100))
        
        results[name] = acc

    print('summary: {:.2f}%'.format(ave_acc / ave_acc_n * 100))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
