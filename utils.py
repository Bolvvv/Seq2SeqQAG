import jieba
import numpy as np

q_file_path = "../data/cMedQA2/question.csv"
a_file_path = "../data/cMedQA2/answer.csv"

q_test_file_path = "q.csv"
a_test_file_path = "a.csv"

q_save_path = "question_trans.csv"
a_save_path = "answer_trans.csv"

def data2list(q_file_path, a_file_path):
    """
    根据数据集的特征，将数据解析为两个list列表，形式均为list[list[]]，按顺序一一对应问与答，具体形式如下
    question: [['头痛', '恶心', '肌肉', '痛', '关节痛', '颈部', '淋巴结', '疼痛', '怎么回事', '啊'], ...]
    answer: [['月经', '延迟', '十四天', '而且', '伴随', '恶心', '，', '头痛', '，', '乏力', '的', '现象', '，', '那么', '考虑', '怀孕', '的', '概率', '是', '非常', '大', '的'],...]
    """
    q_file = open(q_file_path)
    a_file = open(a_file_path)

    #此步将数据均以字典表示，q_map和a_map的key均是question_id
    q_map = dict()
    a_map = dict()
    for q_l in q_file:
        l = q_l.strip('\n').split(',', 1)
        q_map.update({l[0]:l[1]})
    for a_l in a_file:
        l = a_l.strip('\n').split(',', 2)
        a_map.update({l[1]:l[2]})

    #此步对字符串进行切割，并输出目标格式
    q_list = []
    a_list = []
    for i in q_map:
        if a_map.get(i):
            q_list.append(list(jieba.cut(q_map[i])))
            a_list.append(list(jieba.cut(a_map[i])))
    return q_list, a_list#len(q_list) == len(a_list) == 120000

def build_map(q_list, a_list):
    """
    将数据集为对应的id
    返回值为增加了<pad>和<unk>的q2id和a2id的字典
    """
    q2id_map = step_build_map(q_list)
    a2id_map = step_build_map(a_list)

    q2id_map['<unk>'] = len(q2id_map)#UNK用来替代语料中未出现过的单词
    q2id_map['<pad>'] = len(q2id_map)#PAD用来做句子填补，在RNN中需要对sentences的大小进行固定，如果句子短于此，那么就需要填补
    a2id_map['<unk>'] = len(a2id_map)
    a2id_map['<pad>'] = len(a2id_map)

    return q2id_map, a2id_map#len = 60434

def step_build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps