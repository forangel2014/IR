import torch as th
import torch.nn as nn
from transformers import BertConfig, BertModel

class Bert(nn.Module):
    def __init__(self, model_name):
        super(Bert, self).__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        self.linear = nn.Linear(self.config.hidden_size, 1)

    def train(self, mode=True):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    def forward(self, ids, masks):
        output = self.bert(input_ids=ids, attention_mask=masks)[0]
        return th.sigmoid(self.linear(output[:, 0, :]))
