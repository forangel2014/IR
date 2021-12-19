import torch
from torch.utils.data import Dataset
 
class PQDataset(Dataset):

    def __init__(self, id2passages, id2queries, train_triples, gpu_no, tokenizer, max_len, split):
        self.id2passages = id2passages
        self.id2queries = id2queries
        self.train_triples = train_triples
        self.gpu_no = gpu_no
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

    def stat(self, id2passasges):
        N = len(id2passasges) 
        Lave = 0
        DF = {}
        for passage in id2passasges.values():
            ids = self.tokenize(passage, padding=False)
            Lave += ids.shape[0]
            for id in ids:
                if id in DF.keys():
                    DF.update({id, DF[id]+1})
                else:
                    DF.update({id:1})
        Lave /= N
        return N, Lave, DF

    def __len__(self):
        return 2*len(self.train_triples)

    def read_data(self, id2passages, id2queries):
        self.id2passages = id2passages
        self.id2queries = id2queries

    def tokenize(self, text, padding=True):
        ids = self.tokenizer.encode(text, max_length=self.max_len, return_tensors='pt', add_special_tokens=True)
        if not padding:
            return ids
        else:
            pad_ids = torch.zeros([1, self.max_len - ids.shape[1]])
            masks = torch.cat([torch.ones_like(ids).view(1,-1), pad_ids], axis=1)[0].cuda(self.gpu_no)
            ids = torch.cat([ids, pad_ids], axis=1)[0].long().cuda(self.gpu_no)
            return ids, masks

    def tokenize_pq(self, query, passage):
        if self.split:
            ids_query, masks_query = self.tokenize(query)
            ids_passage, masks_passage = self.tokenize(passage)
            return [ids_query, ids_passage], [masks_query, masks_passage]
        else:
            text = query + ' [SEP] ' + passage
            ids_text, masks_text = self.tokenize(text)
            return ids_text, masks_text

    def __getitem__(self, index):
        n = index // 2
        k = index - 2*n
        label = torch.tensor([k]).cuda(self.gpu_no)
        triple = self.train_triples[n]
        query = self.id2queries[triple[0]]
        passage = self.id2passages[triple[2-k]]
        ids, masks = self.tokenize_pq(query, passage)
        return ids, masks, label