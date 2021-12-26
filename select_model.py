import os
sys_names = ['Bert_base', 'DPR', 'BM25']
opt = []
opt_file = './optimal'

for sys_name in sys_names: 
    ls = []
    dir_eval = './eval_result/' + sys_name + '/'
    if not os.path.exists(dir_eval):
        continue
    for filename in os.listdir(dir_eval):
        with open(dir_eval + filename) as f:
            lines = f.readlines()
            for line in lines:
                strs = line.split('\t')
                if 'ndcg_cut_10 ' in strs[0]:
                    val = float(strs[2])
                    ls.append((val, filename))
    res = sorted(ls, key = lambda x: x[0], reverse=True)
    opt.append((sys_name, res[0]))

with open(opt_file, 'w') as f:
    for sys in opt:
        f.writelines(sys[0] + ':\n')
        f.writelines('\tfilename: ' + str(sys[1][1]) + '\n')
        f.writelines('\tndcg@10: ' + str(sys[1][0]) + '\n')
