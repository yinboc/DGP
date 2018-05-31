import argparse
import json

from nltk.corpus import wordnet as wn
import torch

from glove import GloVe


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:
            continue
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='imagenet-split.json')
    parser.add_argument('--output', default='imagenet-induced-graph.json')
    args = parser.parse_args()

    print('making graph ...')

    xml_wnids = json.load(open('imagenet-xml-wnids.json', 'r'))
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)

    js = json.load(open(args.input, 'r'))
    train_wnids = js['train']
    test_wnids = js['test']

    key_wnids = train_wnids + test_wnids

    s = list(map(getnode, key_wnids))
    induce_parents(s, xml_set)

    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)

    wnids = list(map(getwnid, s))
    edges = getedges(s)

    print('making glove embedding ...')

    glove = GloVe('glove.6B.300d.txt')
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {}
    obj['wnids'] = wnids
    obj['vectors'] = vectors.tolist()
    obj['edges'] = edges
    json.dump(obj, open(args.output, 'w'))

