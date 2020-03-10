# coding=utf-8
import os
import pandas as pd
from main import main

if __name__ == "__main__":
    bert_vocab_file = "/home/xyf/models/chinese/bert/pytorch/bert-base-chinese/vocab.txt"
    bert_model_dir = "/home/xyf/models/chinese/bert/pytorch/bert-base-chinese"
    model_name = "BertOrigin"
    output_dir = ".cnews_output/"
    cache_dir = ".cnews_cache/"
    log_dir = ".cnews_log/"


    data_dir = 'data'
    # 从数据集中获取labels,可能会由于数据标签质量问题出错
    df = pd.read_csv(data_dir+'/train.tsv', sep='\t')
    label_list = set(df['label'])
    print(f'label类别数:{len(label_list)},{label_list}')

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == 'BertLSTM':
        from BertLSTM import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args

    elif model_name == "BertDPCNN":
        from BertDPCNN import args
    
    config = args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir)
    main(config, config.save_name, label_list)

