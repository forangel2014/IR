import torch as th
import torch.nn as nn
from transformers import BertConfig, BertModel

class DPR(nn.Module):
    def __init__(self, model_name):
        super(DPR, self).__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert_query = BertModel.from_pretrained(model_name)
        self.bert_passage = BertModel.from_pretrained(model_name)

    def train(self, mode=True):
        self.bert_query.train()
        self.bert_passage.train()

    def eval(self):
        self.bert_query.eval()
        self.bert_passage.eval()

    def forward(self, ids, masks):
        output_query = self.bert_query(input_ids=ids[0], attention_mask=masks[0])[0][:, 0, :]
        output_passage = self.bert_passage(input_ids=ids[1], attention_mask=masks[1])[0][:, 0, :]

        batch_size = output_query.shape[0]
        dim = output_query.shape[1]
        Eq = output_query/th.norm_except_dim(output_query)
        Ep = output_passage/th.norm_except_dim(output_passage)
        Eq = Eq.view(batch_size, 1, dim)
        Ep = Ep.view(batch_size, dim, 1)
        return th.matmul(Eq, Ep)[:, :, 0]
