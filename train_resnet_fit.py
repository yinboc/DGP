import argparse
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import set_gpu, ensure_path
from models.resnet import ResNet
from datasets.image_folder import ImageFolder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred')
    parser.add_argument('--train-dir')
    parser.add_argument('--save-path', default='save/resnet-fit')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    set_gpu(args.gpu)
    save_path = args.save_path
    ensure_path(save_path)

    pred = torch.load(args.pred)
    train_wnids = sorted(os.listdir(args.train_dir))

    train_dir = args.train_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))
    loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)

    assert pred['wnids'][:1000] == train_wnids

    model = ResNet('resnet50', 1000)
    sd = model.resnet_base.state_dict()
    sd.update(torch.load('materials/resnet50-base.pth'))
    model.resnet_base.load_state_dict(sd)

    fcw = pred['pred'][:1000].cpu()
    model.fc.weight = nn.Parameter(fcw[:, :-1])
    model.fc.bias = nn.Parameter(fcw[:, -1])

    model = model.cuda()
    model.train()

    optimizer = torch.optim.SGD(model.resnet_base.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().cuda()
    
    keep_ratio = 0.9975
    trlog = {}
    trlog['loss'] = []
    trlog['acc'] = []

    for epoch in range(1, 9999):

        ave_loss = None
        ave_acc = None

        for i, (data, label) in enumerate(loader, 1):
            data = data.cuda()
            label = label.cuda()

            logits = model(data)
            loss = loss_fn(logits, label)

            _, pred = torch.max(logits, dim=1)
            acc = torch.eq(pred, label).type(torch.FloatTensor).mean().item()

            if i == 1:
                ave_loss = loss.item()
                ave_acc = acc
            else:
                ave_loss = ave_loss * keep_ratio + loss.item() * (1 - keep_ratio)
                ave_acc = ave_acc * keep_ratio + acc * (1 - keep_ratio)

            print('epoch {}, {}/{}, loss={:.4f} ({:.4f}), acc={:.4f} ({:.4f})'
                  .format(epoch, i, len(loader), loss.item(), ave_loss, acc, ave_acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trlog['loss'].append(ave_loss)
        trlog['acc'].append(ave_acc)

        torch.save(trlog, osp.join(save_path, 'trlog'))

        torch.save(model.resnet_base.state_dict(),
                   osp.join(save_path, 'epoch-{}.pth'.format(epoch)))

