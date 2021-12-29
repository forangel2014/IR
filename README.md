# 信息检索大作业：TREC 2020 Deep Learning Passage Ranking
## 文件说明：
+ ./data: 数据目录
+ ./models: 模型代码目录
    + bert.py: BERT检索模型
    + bm25.py: BM25检索模型
    + dpr.py: DPR检索模型
+ ./utils: 工具脚本目录
    + arguments.py: 参数
    + dataset.py: 准备数据集
    + eval.py: 对重排结果进行评测
    + select_model.py: 选择验证集评测结果中最优的模型
+ ./valid_results/: 验证集重排文件目录
    + ./eval/: 验证集重排结果评测目录
+ ./test_results/: 测试集重排文件目录
    + ./eval/: 测试集重排结果评测目录
+ ./saves/: 模型训练文件目录
+ main.py: 训练主程序
+ irsystem.py: 系统模块
+ optimal: 验证集最优模型信息
+ requirements.txt: 运行依赖包列表

## 运行流程：
1. 选择超参数及使用的模型，运行main.py，重排结果将存储在./valid_results/的对应目录下
2. 运行eval.py调用trec_eval脚本评测各重排结果（需要通过<https://trec.nist.gov/trec_eval/>下载trec_eval到当前目录下，如./trec_eval-9.0.7，并进入其中执行make命令）
3. 运行select_model.py选取评测结果最优的模型，写入optimal中