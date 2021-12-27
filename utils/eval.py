import os

def eval(test_dir='./test_results/BM25/', eval_dir='./eval_results/BM25/'):
    for filename in os.listdir(test_dir):
        os.system('./trec_eval-9.0.7/trec_eval -m ndcg_cut ./data/2020qrels-pass.txt ' +
                    test_dir + filename + ' > ' + eval_dir + filename)
    print('evaluation completed.')