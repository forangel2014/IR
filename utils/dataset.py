import torch as th
from torch.utils.data import Dataset

class PQdataset(Dataset):
    def __init__(self, id2passages, id2queries, train_triples, device, tokenizer, max_len, split):
        self.id2passages = id2passages
        self.id2queries = id2queries
        self.train_triples = train_triples
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

    def stat(self, id2passages):
        n = len(id2passages)
        l_ave = 0
        dfs = {}
        for passage in id2passages.values():
            ids = self.tokenize(passage, padding=False)[0].tolist()[0]
            l_ave += len(ids)
            ids = list(set(ids))
            for id in ids:
                if id in dfs.keys():
                    dfs.update({id: dfs[id] + 1})
                else:
                    dfs.update({id: 1})
        l_ave /= n
        return n, l_ave, dfs

    def __len__(self):
        return 2 * len(self.train_triples)

    def read_data(self, id2passages, id2queries):
        self.id2passages = id2passages
        self.id2queries = id2queries

    def tokenize(self, text, padding=True):
        ids = self.tokenizer.encode(text, max_length=self.max_len, return_tensors='pt', add_special_tokens=True)
        if not padding:
            return ids, None
        else:
            pad_ids = th.zeros([1, self.max_len - ids.shape[1]])
            masks = th.cat([th.ones_like(ids).view(1, -1), pad_ids], dim=1)[0].cuda(self.device)
            ids = th.cat([ids, pad_ids], dim=1)[0].long().cuda(self.device)
            return ids, masks

    def tokenize_pq(self, query, passage, padding=True):
        if self.split:
            ids_query, masks_query = self.tokenize(query, padding)
            ids_passage, masks_passage = self.tokenize(passage, padding)
            return [ids_query, ids_passage], [masks_query, masks_passage]
        else:
            text = query + ' [SEP] ' + passage
            ids_text, masks_text = self.tokenize(text, padding)
            return ids_text, masks_text

    def __getitem__(self, index):
        n = index // 2
        k = index - 2 * n
        label = th.tensor([k]).cuda(self.device)
        triple = self.train_triples[n]
        query = self.id2queries[triple[0]]
        passage = self.id2passages[triple[2 - k]]
        ids, masks = self.tokenize_pq(query, passage)
        return ids, masks, label
