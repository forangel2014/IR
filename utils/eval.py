import os

def eval(test_dir='./test_results/', eval_dir='./eval_results/', sys_names=['Bert', 'DPR', 'BM25']):
    for sys_name in sys_names:
        test_sys_dir = test_dir + sys_name + '/'
        eval_sys_dir = eval_dir + sys_name + '/'
        for filename in os.listdir(test_sys_dir):
            os.system('./trec_eval-9.0.7/trec_eval -m ndcg_cut ./data/2020qrels-pass.txt ' +
                        test_sys_dir + filename + ' > ' + eval_sys_dir + filename)
        print('evaluation completed.')