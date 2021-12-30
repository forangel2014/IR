# 信息检索大作业：TREC 2020 Deep Learning Passage Ranking

## 成员信息
+ 组长：孙望涛 202128014628017
+ 金卓然
+ 徐遥
+ 杨晨

## 实验方案

### BM25模型
直接基于课堂所讲授的BM25公式实现

### BERT模型
预训练后的BERT具有强大的语言特征抽取能力，我们将Query与Passage拼接，送入BERT中，将其输出层的[CLS]位置处的表示向量作为整体的表征，经过线性层后得到标量值，代表
该Query与Passage的匹配得分score，考虑到训练集只标注了正负例，我们将score经过sigmoid函数得到0-1间的标量，再与标签求交叉熵作为损失函数。


### DPR模型

## 文件说明
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

## 运行方式
直接运行main.py即可，附加参数见./utils/arguments.py

## 运行流程
1. 解析所选择的超参数及使用的模型
2. 系统生成IRsystem对象，对数据集进行解析，获得训练的正负例样本与待重排的top1000数据
3. 根据所选模型，生成一系列控制参数，载入预训练的BERT模型，生成dataloader，优化器等模块
4. 开始训练，每个epoch结束时会存储模型文件到对应目录下(默认在./saves/sys_name中)，并且在验证集上进行一次ranking，生成标准的TREC格式文件(默认在./valid_results/sys_name/中)
5. 所有epoch训练完成后，调用eval函数（此函数调用trec_eval脚本，需要预先下载并make，路径应为./trec_eval-9.0.7/trec_eval），将评测结果写入对应目录中(默认在./valid_results/eval/sys_name中)
6. 调用select_model函数，从验证集的评测结果中挑选出最优的模型，返回其文件名
7. 系统根据文件名载入对应的模型文件，在测试集上进行ranking，生成标准的TREC格式文件(默认在./test_results/eval/sys_name中)
8. 调用eval函数，评测测试结果，写入对应目录(默认在./test_results/eval/sys_name中)

注：代表测试集的test_top_file在irsystem.py中只被调用一次（测试时），明显表明我们没有用其进行训练

## 实验结果