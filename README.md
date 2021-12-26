# 信息检索大作业：TREC 2020 Deep Learning Passage Ranking
## 文件说明：
+ ./data: 数据目录
+ ./test_result/: 重排文件目录
+ ./eval_result/: 评测结果目录
+ main.py: 训练主程序
+ dataset.py: 准备数据集
+ model.py: 模型实现
+ eval.py: 对重排结果进行评测
+ select_model.py: 选择评测结果中最优的模型

## 运行流程：
1. 选择超参数及使用的模型，运行main.py，重排结果将存储在./test_result/的对应目录下
2. 运行eval.py调用trec_eval脚本评测各重排结果（需要通过<https://trec.nist.gov/trec_eval/>下载trec_eval到当前目录下，如./trec_eval-9.0.7，并进入其中执行make命令）
3. 运行select_model.py选取评测结果最优的模型