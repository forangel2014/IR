import os
#sys_name = 'Bert_base'
#sys_name = 'DPR'
sys_name = 'BM25'
dir_test = './test_result/' + sys_name + '/'
dir_eval = './eval_result/' + sys_name + '/'

if not os.path.exists(dir_eval):
    os.mkdir(dir_eval)
for f in os.listdir(dir_test):
    #print(f)
    os.system('./trec_eval-9.0.7/trec_eval -m ndcg_cut 2020qrels-pass.txt ' + dir_test + f + ' > ' + dir_eval + f)