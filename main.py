from dataset import PQDataset
from torch.utils.data import DataLoader
from model import *
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

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

id2passages = parse_pq_file(train_passages_file)
id2queries = parse_pq_file(train_queries_file)
train_triples = parse_triple_file(train_triples_file)

model_name = 'bert-base-uncased'
bert_config = BertConfig.from_pretrained(model_name)
#model = BertForSequenceClassification.from_pretrained(model_name).cuda()
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset = PQDataset(id2passages, id2queries, train_triples, tokenizer, max_len=160)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

model = BertScoringModel(model_name).cuda()
#model.bert.train()
optmizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i, [ids, labels] in enumerate(dataloader):
    optmizer.zero_grad()
    ids = ids.long().cuda()
    labels = labels.cuda()
    #output = model(ids, labels)[0]
    output = model(ids)
    #loss = torch.sum(output[:,0])
    #loss = -torch.sum(labels*torch.log(output)+(1-labels)*torch.log(1-output))
    loss = torch.sum((labels-output)**2)
    loss.backward()
    print(loss)
    optmizer.step()

print("finish training")

model.bert.eval()
id2queries, id2passages, qid2pid = parse_top1000_file(validation_top_file)
model.read_data(id2passages, id2queries)
for qid in qid2pid.keys():
    for pid in qid2pid[qid]:
        with torch.no_grad():
            loss, _ = model([qid, pid, 1])
            score = -loss
            print(score)
