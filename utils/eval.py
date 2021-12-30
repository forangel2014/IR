import os

def eval(valid_dir='./valid_results/', qrels_file = './data/2019qrels-pass.txt', sys_names=['Bert', 'DPR', 'BM25']):
    eval_dir = valid_dir + 'eval/'
    os.makedirs(eval_dir, exist_ok=True)
    for sys_name in sys_names:
        valid_sys_dir = valid_dir + sys_name + '/'
        eval_sys_dir = eval_dir + sys_name + '/'
        os.makedirs(eval_sys_dir, exist_ok=True)
        if os.path.exists(valid_sys_dir):
            for filename in os.listdir(valid_sys_dir):
                os.system('./trec_eval-9.0.7/trec_eval -m ndcg_cut ' + qrels_file + ' ' +
                            valid_sys_dir + filename + ' > ' + eval_sys_dir + filename)
            print('evaluation completed.')