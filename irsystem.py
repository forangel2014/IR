from models.bm25 import BM25
from models.dpr import DPR
from models.bert import Bert
from utils.arguments import get_common_args
from utils.dataset import PQdataset
from utils.parser import *
from utils.eval import *
import os
import random
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

class IRSystem(object):
    def __init__(self, args):
        self.args = args
        self.train_passages_file = args.data_dir + 'collection.train.sampled.tsv'  # qid-query
        self.train_queries_file = args.data_dir + 'queries.train.sampled.tsv'  # pid-passage
        self.train_triples_file = args.data_dir + 'qidpidtriples.train.sampled.tsv'  # qid-pos_id-neg_id
        self.validation_top_file = args.data_dir + 'msmarco-passagetest2019-43-top1000.tsv'  # qid-pid-query-passage
        self.validation_qrels_file = args.data_dir + '2019qrels-pass.txt'  # qid-"Q0"-pid-rating
        self.test_top_file = args.data_dir + 'msmarco-passagetest2020-54-top1000.tsv'  # qid-pid-query-passage
        self.test_qrels_file = args.data_dir + '2020qrels-pass.txt'  # qid-"Q0"-pid-rating

        self.test_dir = args.test_dir + args.sys_name + '/'
        self.eval_dir = args.eval_dir + args.sys_name + '/'
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.split_pq = not (args.sys_name == 'Bert')
        self.padding = not (args.sys_name == 'BM25')
        self.need_opt = not (args.sys_name == 'BM25')

        id2passages = parse_pq_file(self.train_passages_file)
        id2queries = parse_pq_file(self.train_queries_file)
        train_triples = parse_triple_file(self.train_triples_file)

        bert_config = BertConfig.from_pretrained(args.model_name)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        self.dataset = PQdataset(id2passages, id2queries, train_triples, args.device, tokenizer,
                            max_len=args.max_len, split=self.split_pq)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)

        if args.sys_name == 'Bert':
            self.model = Bert(args.model_name)
        elif args.sys_name == 'DPR':
            self.model = DPR(args.model_name)
        elif args.sys_name == 'BM25':
            n, l_ave, dfs = self.dataset.stat(id2passages)
            self.model = BM25(k1=args.k1, k3=args.k3, b=args.b, n=n, l_ave=l_ave, dfs=dfs)

        if self.need_opt:
            self.model = self.model.cuda(self.args.device)
            self.optimizer = Adam(self.model.parameters(), lr=args.lr)

    def train(self):
        self.model.train()
        for i, [ids, masks, labels] in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            output = self.model(ids, masks)
            loss = -th.sum(labels * th.log(output) + (1 - labels) * th.log(1 - output))
            loss.backward()
            self.optimizer.step()
            print('loss: {}'.format(loss.item()))
        print('training completed.')

    def test(self, info):
        self.model.eval()
        print('start testing.')
        id2queries, id2passages, qid2pid = parse_top1000_file(self.test_top_file)
        with open(self.test_dir + info + self.args.result_file, 'w') as f:
            for qid in qid2pid.keys():
                pid2score = {}
                for pid in qid2pid[qid]:
                    query = id2queries[qid]
                    passage = id2passages[pid]
                    ids, masks = self.dataset.tokenize_pq(query, passage, padding=self.padding)
                    if self.split_pq:
                        ids = [x.view(1, -1) for x in ids]
                        if self.padding:
                            masks = [x.view(1, -1) for x in masks]
                    else:
                        ids = ids.view(1, -1)
                        if self.padding:
                            masks = masks.view(1, -1)
                    with th.no_grad():
                        score = self.model(ids, masks).squeeze().detach().cpu().numpy()
                        pid2score.update({pid: score})
                res = sorted(pid2score.items(), key=lambda x: x[1], reverse=True)
                for i in range(len(res)):
                    pid = res[i][0]
                    score = res[i][1]
                    f.writelines(str(qid) + ' ' + 'Q0' + ' ' + str(pid) + ' ' +
                                 str(i) + ' ' + str(score) + ' ' + self.args.sys_name + '\n')
        print('testing completed.')

    def run(self):
        if self.need_opt:
            for epoch in range(self.args.epoches):
                print('training epoch {} in total {} epoches'.format(epoch, self.args.epoches))
                self.train()
                info = 'epoch_{}_lr_{}_batchsize_{}_'.format(epoch, self.args.lr, self.args.batch_size)
                self.test(info)
        else:
            for epoch in range(self.args.epoches):
                random.seed()
                k1 = random.random()
                k3 = random.random()
                b = random.random()
                info = 'k1_{:.2f}_k3_{:.2f}_b_{:.2f}_'.format(k1, k3, b)
                print("para search: " + info)
                self.model.set_paras(k1, k3, b)
                self.test(info)
        eval(self.test_dir, self.eval_dir)

if __name__ == '__main__':
    args = get_common_args()
    #ir_sys = IRSystem(args)
    #ir_sys.run()
    eval()