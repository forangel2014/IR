import torch as th
import torch.nn as nn
import math

class BM25(nn.Module):
    def __init__(self, k1, k3, b, n, l_ave, dfs):
        super(BM25, self).__init__()
        self.k1 = k1
        self.k3 = k3
        self.b = b
        self.n = n
        self.l_ave = l_ave
        self.dfs = dfs

    def set_paras(self, k1, k3, b):
        self.k1 = k1
        self.k3 = k3
        self.b = b        

    def forward(self, ids, masks):
        scores = 0
        query_ids = ids[0].tolist()[0]
        passage_ids = ids[1].tolist()[0]
        qids = list(set(query_ids))

        for qid in qids:
            qtf = query_ids.count(qid)
            ptf = passage_ids.count(qid)
            l_d = len(passage_ids)
            df = 0
            if qid in self.dfs.keys():
                df = self.dfs[qid]

            term1 = qtf / (self.k3 + qtf)
            term2 = self.k1 * ptf / (ptf + self.k1 * (1 - self.b + self.b * l_d / self.l_ave))
            term3 = math.log2((self.n - df + 0.5) / (df + 0.5))
            score = term1 * term2 * term3

            scores += score

        return th.tensor([scores])
