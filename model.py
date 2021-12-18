import torch
from transformers import BertConfig, BertModel

class BertScoringModel(torch.nn.Module):
    
    def __init__(self, model_name):

        super(BertScoringModel, self).__init__()
        self.model_name = model_name
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(self, ids, masks):
        output = self.bert(input_ids=ids, attention_mask=masks)[0]
        return torch.sigmoid(self.linear(output[:,0,:]))