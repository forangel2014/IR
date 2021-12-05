import torch
from torch.utils.data import Dataset
 
class PQDataset(Dataset):

    def __init__(self, id2passages, id2queries, train_triples, tokenizer, max_len = 160):
        self.id2passages = id2passages
        self.id2queries = id2queries
        self.train_triples = train_triples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return 2*len(self.train_triples)

    def read_data(self, id2passages, id2queries):
        self.id2passages = id2passages
        self.id2queries = id2queries

    def __getitem__(self, index):
        n = index // 2
        k = index - 2*n
        triple = self.train_triples[n]
        query = self.id2queries[triple[0]]
        passage = self.id2passages[triple[2-k]]
        text = '[CLS] ' + query + ' [SEP] ' + passage
        ids = self.tokenizer.encode(text, max_length=self.max_len, return_tensors='pt')
        pad_ids = torch.zeros([1, self.max_len - ids.shape[1]])
        ids = torch.cat([ids, pad_ids], axis=1)[0]
        return ids, torch.tensor([k])