import argparse

def get_common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/', help='data directory')
    parser.add_argument('--valid_dir', type=str, default='./valid_results/', help='result valid directory')
    parser.add_argument('--eval_dir', type=str, default='./eval_results/', help='result eval directory')
    parser.add_argument('--save_dir', type=str, default='./saves/', help='model saved directory')
    parser.add_argument('--result_file', type=str, default='results.txt', help='result file name')
    parser.add_argument('--sys_name', type=str, default='Bert', help='IR model name')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Bert model name')
    parser.add_argument('--device', type=int, default=0, help='GPU device number')

    parser.add_argument('--max_len', type=int, default=160, help='maximum sentence length')
    parser.add_argument('--epoches', type=int, default=100, help='training epoch number')
    parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

    parser.add_argument('--k1', type=float, default=1, help='k1 in BM25 formula')
    parser.add_argument('--k3', type=float, default=1, help='k3 in BM25 formula')
    parser.add_argument('--b', type=float, default=1, help='b in BM25 formula')

    args = parser.parse_args()
    return args