import torch
from transformers import BertConfig, BertModel

class BertScoringModel(torch.nn.Module):
    
    def __init__(self, model_name):

        super(BertScoringModel, self).__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.config.hidden_size, 1)

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def forward(self, ids, masks):
        output = self.bert(input_ids=ids, attention_mask=masks)[0]
        return torch.sigmoid(self.linear(output[:,0,:]))

class BM25ScoringModel(torch.nn.Module):
    
    def __init__(self, k1, k2, k3, b, N, Lave, DF):

        super(BM25ScoringModel, self).__init__()
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.b = b
        self.N = N
        self.Lave = Lave
        self.DF = DF

    #def stat(self, id2passages, id2queries):

    def forward(self, ids, masks):
        score = torch.Tensor(0)
        query_ids = ids[0]
        passage_ids = ids[1]
        for id in query_ids:
            qtf = torch.sum(query_ids == id)
            ptf = torch.sum(passage_ids == id)
            Ld = passage_ids.shape[1]
            score += (qtf/(self.k3+qtf))* \
                     (self.k1*ptf/(ptf+self.k1*(1-self.b+self.b*Ld/self.Lave)))* \
                     (torch.log2((self.N - self.DF(id)+0.5)/(self.DF(id)+0.5)))
        return torch.sigmoid(score)

class DPRScoringModel(torch.nn.Module):
    
    def __init__(self, model_name):

        super(DPRScoringModel, self).__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert_query = BertModel.from_pretrained(model_name)
        self.bert_passage = BertModel.from_pretrained(model_name)

    def train(self):
        self.bert_query.train()
        self.bert_passage.train()

    def eval(self):
        self.bert_query.eval()
        self.bert_passage.eval()

    def forward(self, ids, masks):
        output_query = self.bert_query(input_ids=ids[0], attention_mask=masks[0])[0][:,0,:]
        output_passage = self.bert_passage(input_ids=ids[1], attention_mask=masks[1])[0][:,0,:]
        batchsize = output_query.shape[0]
        dim = output_query.shape[1]
        a = output_query.view(batchsize,1,dim)
        b = output_passage.view(batchsize,dim,1)
        return torch.sigmoid(torch.matmul(a, b)[:,:,0]*1e-3)