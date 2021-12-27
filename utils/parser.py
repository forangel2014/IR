def parse_pq_file(filename):
    dict = {}
    with open(filename, 'r', encoding='utf-8-sig') as f_input:
        for line in f_input:
            id, text = list(line.strip().split('\t'))
            dict.update({id: text})
    return dict

def parse_triple_file(filename):
    ls = []
    with open(filename, 'r', encoding='utf-8-sig') as f_input:
        for line in f_input:
            ls.append(list(line.strip().split('\t')))
    return ls

def parse_top1000_file(filename):
    id2queries = {}
    id2passages = {}
    qid2pid = {}
    with open(filename, 'r', encoding='utf-8-sig') as f_input:
        for line in f_input:
            qid, pid, query, passage = list(line.strip().split('\t'))
            if not (qid in id2queries.keys()):
                id2queries.update({qid: query})
            if not (pid in id2passages.keys()):
                id2passages.update({pid: passage})
            if not (qid in qid2pid.keys()):
                qid2pid.update({qid: [pid]})
            else:
                qid2pid[qid].append(pid)
    return id2queries, id2passages, qid2pid
