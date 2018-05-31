import torch


class GloVe():

    def __init__(self, file_path):
        self.dimension = None
        self.embedding = dict()
        with open(file_path, 'r') as f:
            for line in f.readlines():
                strs = line.strip().split()
                word = strs[0]
                vector = torch.FloatTensor(list(map(float, strs[1:])))
                self.embedding[word] = vector
                if self.dimension is None:
                    self.dimension = len(vector)

    def _fix_word(self, word):
        terms = word.replace('_', ' ').split(' ')
        ret = self.zeros()
        cnt = 0
        for term in terms:
            v = self.embedding.get(term)
            if v is None:
                subterms = term.split('-')
                subterm_sum = self.zeros()
                subterm_cnt = 0
                for subterm in subterms:
                    subv = self.embedding.get(subterm)
                    if subv is not None:
                        subterm_sum += subv
                        subterm_cnt += 1
                if subterm_cnt > 0:
                    v = subterm_sum / subterm_cnt
            if v is not None:
                ret += v
                cnt += 1
        return ret / cnt if cnt > 0 else None

    def __getitem__(self, words):
        if type(words) is str:
            words = [words]
        ret = self.zeros()
        cnt = 0
        for word in words:
            v = self.embedding.get(word)
            if v is None:
                v = self._fix_word(word)
            if v is not None:
                ret += v
                cnt += 1
        if cnt > 0:
            return ret / cnt
        else:
            return self.zeros()
    
    def zeros(self):
        return torch.zeros(self.dimension)
 
