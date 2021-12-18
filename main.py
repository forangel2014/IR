import matplotlib.pyplot as plt
import numpy as np
from dataset import PQDataset
from torch.utils.data import DataLoader
from model import *
import os
from transformers import BertTokenizer, BertConfig

train_passages_file = 'collection.train.sampled.tsv' # qid - query
train_queries_file = 'queries.train.sampled.tsv' # pid - passage
train_triples_file = 'qidpidtriples.train.sampled.tsv' # qid - pos_id - neg_id
validation_top_file = 'msmarco-passagetest2019-43-top1000.tsv' # qid - pid - query - passage
validation_qrels_file = '2019qrels-pass.txt' # qid - "Q0" - pid - rating
test_top_file = 'msmarco-passagetest2020-54-top1000.tsv' # qid - pid - query - passage
test_qrels_file = '2020qrels-pass.txt' # qid - "Q0" - pid - rating

def parse_pq_file(filename):
    dict = {}
    with open(filename,'r',encoding='utf-8-sig') as f_input:
        for line in f_input:
            id, text = list(line.strip().split('\t'))
            dict.update({id:text})
    return dict

def parse_triple_file(filename):
    ls = []
    with open(filename,'r',encoding='utf-8-sig') as f_input:
        for line in f_input:
            ls.append(list(line.strip().split('\t')))
    return ls

def parse_top1000_file(filename):
    id2queries = {}
    id2passages = {}
    qid2pid = {}
    with open(filename,'r',encoding='utf-8-sig') as f_input:
        for line in f_input:
            qid, pid, query, passage = list(line.strip().split('\t'))
            if not qid in id2queries.keys():
                id2queries.update({qid:query})
            if not pid in id2passages.keys():
                id2passages.update({pid:passage})
            if not qid in qid2pid.keys():
                qid2pid.update({qid:[pid]})
            else:
                qid2pid[qid].append(pid)
    return id2queries, id2passages, qid2pid

def plot_curve(res):
    plt.figure()
    steps = np.array(range(len(res)))
    plt.plot(steps, res)
    plt.show()

id2passages = parse_pq_file(train_passages_file)
id2queries = parse_pq_file(train_queries_file)
train_triples = parse_triple_file(train_triples_file)

model_name = 'bert-base-uncased'
bert_config = BertConfig.from_pretrained(model_name)
#model = BertForSequenceClassification.from_pretrained(model_name).cuda()
tokenizer = BertTokenizer.from_pretrained(model_name)

#hyper-parameters
max_len = 160
batch_size = 10
epoch = 10
lr = 1e-5
skip_train = False
sys_name = 'Bert_base'
dir_out = './test_result/' + sys_name + '/'
res_file = 'res20'
os.makedirs(dir_out, exist_ok=True)

dataset = PQDataset(id2passages, id2queries, train_triples, tokenizer, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BertScoringModel(model_name).cuda()
optmizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_list = []

if not skip_train:
    for e in range(epoch):
        model.bert.train()
        for i, [ids, masks, labels] in enumerate(dataloader):
            optmizer.zero_grad()
            ids = ids.long().cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            output = model(ids, masks)
            loss = -torch.sum(labels*torch.log(output)+(1-labels)*torch.log(1-output))
            #loss = torch.sum((labels-output)**2)
            loss.backward()
            optmizer.step()
            print(loss)
            loss_list.append(loss.detach().cpu().numpy())

        print("finish training epoch " + str(e))

        model.bert.eval()
        id2queries, id2passages, qid2pid = parse_top1000_file(test_top_file)
        with open(dir_out + res_file + '_epoch' + str(e), 'w') as f:
            for qid in qid2pid.keys():
                pid2score = {}
                for pid in qid2pid[qid]:
                    query = id2queries[qid]
                    passage = id2passages[pid]
                    text = query + ' [SEP] ' + passage
                    ids = tokenizer.encode(text, max_length=max_len, return_tensors='pt', add_special_tokens=True)
                    pad_ids = torch.zeros([1, max_len - ids.shape[1]])
                    masks = torch.cat([torch.ones_like(ids).view(1,-1), pad_ids], axis=1).cuda()
                    ids = torch.cat([ids, pad_ids], axis=1).long().cuda()
                    with torch.no_grad():
                        score = model(ids, masks)[0][0].detach().cpu().numpy()
                        print(score)
                        pid2score.update({pid:score})
                res = sorted(pid2score.items(), key=lambda x:x[1], reverse=True)
                for i in range(len(res)):
                    pid = res[i][0]
                    score = res[i][1]
                    f.writelines(str(qid) + ' ' + 'Q0' + ' ' + str(pid) + ' ' + str(i) + ' ' + str(score) + ' ' + sys_name + '\n')
    
