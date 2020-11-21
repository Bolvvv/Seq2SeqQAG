import jieba
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split

q_file_path = "../data/cMedQA2/question.csv"
a_file_path = "../data/cMedQA2/answer.csv"

def data2list(q_file_path, a_file_path):
    """
    根据数据集的特征，将数据解析为两个list列表，形式均为list[list[]]，按顺序一一对应问与答，具体形式如下
    q_list: [['头痛', '恶心', '肌肉', '痛', '关节痛', '颈部', '淋巴结', '疼痛', '怎么回事', '啊'], ...]
    a_list: [['月经', '延迟', '十四天', '而且', '伴随', '恶心', '，', '头痛', '，', '乏力', '的', '现象', '，', '那么', '考虑', '怀孕', '的', '概率', '是', '非常', '大', '的'],...]
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

def pad_batch_word_to_id(q_list, a_list, pad_token, q_map, a_map, device):
    """
    将形如list[list[]]的q和a(其中值为中文)转换为list[list[]]的q和a(其中值为字典中的对应数字)，
    且从高到低进行排序，再进行数据填充，最后返回tensor化的值
    """
    #将中文转换为对应map的id
    q_trans = word2id(q_list, q_map)
    a_trans = word2id(a_list, a_map)
    
    #对列表进行填充，并存储其排序后的每个句子的长度
    q_trans_pad, q_len = pad_list(q_trans, pad_token)
    a_trans_pad, a_len = pad_list(a_trans, pad_token)

    #转换为float类型并将其传入到gpu中
    q_tensor = torch.tensor(q_trans_pad, dtype=torch.long, device=device)
    a_tensor = torch.tensor(a_trans_pad, dtype=torch.long, device=device)
    return q_tensor, a_tensor, q_len, a_len

def word2id(lists, word_map):
    """
    将形如list[list[]]其中的值为中文转换为对应map的id，最终返回List[listp[]]，其中值为数字id
    """
    for l in lists:
        for i in range(len(l)):
            l[i] = word_map[l[i]]
    return l

def pad_list(lists, pad_token):
    """
    将形如list[list[]]按照里面list的长度从高到底排列，并进行填充使其长度一致，最后返回形如list[listp[]]，
    其中值为数字，以及该列表的原始长度
    """
    sents_padded = []
    lists.sort(key=lambda l: len(l),reverse=True)

    lists_len = [len(s) for s in lists]#存储原始长度

    max_length = len(lists[0])
    for l in lists:
        while len(l) != max_length:
            l.append(pad_token)
    sents_padded = lists
    return sents_padded, lists_len

def cut_train_dev_test_data(q_list, a_list):
    """
    在此步进行数据切割，目前暂时将数据按照顺序进行切割
    ratio_train = 0.6
    ratio_dev = 0.2
    ratio_test = 0.2
    """

    q_train, q_tmp, a_train, a_tmp = train_test_split(q_list, a_list, test_size=0.4, random_state=1)
    q_dev, q_test, a_dev, a_test = train_test_split(q_tmp, a_tmp, test_size=0.5, random_state=1)

    return q_train, a_train, q_dev, a_dev, q_test, a_test

def save_data(q_train, a_train, q_dev, a_dev, q_test, a_test):
    q_train = np.array(q_train)
    np.save('./data/q_train.npy', q_train)

    a_train = np.array(a_train)
    np.save('./data/a_train.npy', a_train)

    q_dev = np.array(q_dev)
    np.save('./data/q_dev.npy', q_dev)

    a_dev = np.array(a_dev)
    np.save('./data/a_dev.npy', a_dev)

    q_test = np.array(q_test)
    np.save('./data/q_test.npy', q_test)

    a_test = np.array(a_test)
    np.save('./data/a_test.npy', a_test)

def load_data(file_path):
    data=np.load(file_path, allow_pickle=True)
    data=data.tolist()
    return data

# q_list, a_list = data2list(q_file_path, a_file_path)
# q2id_map, a2id_map = build_map(q_list, a_list)
# pad_batch_word_to_id(q_list, a_list, 0,q2id_map, a2id_map)
# q_train, a_train, q_dev, a_dev, q_test, a_test = cut_train_dev_test_data(q_list, a_list)

# save_data(q_train, a_train, q_dev, a_dev, q_test, a_test)

# print(type(load_data("./data/q_train.npy")[13]))
# print(load_data("./data/a_train.npy")[13])