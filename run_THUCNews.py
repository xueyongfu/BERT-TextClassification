# coding=utf-8
from main import main

if __name__ == "__main__":

    model_name = "BertLSTM"
    label_list = [u'房产', u'科技', u'财经', u'游戏',u'彩票', u'股票', u'社会', u'星座',
                  u'娱乐', u'时尚', u'时政', u'家居', u'教育', u'体育']
    data_dir = "/home/xyf/桌面/Disk/文本分类数据集/THUCNews/small_THUCNews"
    output_dir = ".cnews_output/"
    cache_dir = ".cnews_cache/"
    log_dir = ".cnews_log/"

    # model_times = "model_1/"   # 第几次保存的模型，主要是用来获取多次最佳结果

    bert_vocab_file = "/home/xyf/models/chinese/bert/pytorch/bert-base-chinese/vocab.txt"  # 需改
    bert_model_dir = "/home/xyf/models/chinese/bert/pytorch/bert-base-chinese"  

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

