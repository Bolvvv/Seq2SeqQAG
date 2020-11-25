import utils
import torch
from my_model import my_model
import sys
import time
import math
import numpy as np

BATCH_SIZE = 10 #仅能设置为10的倍数
EMB_SIZE = 300 #词嵌入维度
HIDDEN_SIZE = 300 #隐藏层
LR = 0.001 #学习率
EPOCH = 5 #进行多少个epoch后停下来
model_save_path = 'my_model.pkl'

def evaluate_ppl(model, dev_q, dev_a, batch_size):
    was_training = model.training
    model.eval()

    cum_loss=0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for ind in range(0, len(dev_q), batch_size):
            batch_q = dev_q[ind:ind+BATCH_SIZE]
            batch_a = dev_a[ind:ind+BATCH_SIZE]

            loss = -model(batch_q, batch_a).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in batch_a)
            cum_tgt_words += tgt_word_num_to_predict
        ppl = np.exp(cum_loss / cum_tgt_words)
    if was_training:
        model.train()

    return ppl

def train():
    #数据导入
    train_q = utils.load_data("./data/q_train.npy")
    train_a = utils.load_data("./data/a_train.npy")
    dev_q = utils.load_data("./data/q_dev.npy")
    dev_a = utils.load_data("./data/a_dev.npy")
    
    train_batch_size = BATCH_SIZE

    #生成字典
    q2id_map, a2id_map = utils.build_map(train_q+dev_q, train_a+dev_a)

    #设置梯度裁剪
    clip_grad = 5.0

    #生成模型
    model = my_model(q2id_map, a2id_map, EMB_SIZE, HIDDEN_SIZE, dropout_rate=0.2)
    model.train()

    #TODO:进行Uniform操作

    #TODO:进行掩码设置

    #GPU选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('use device: %s' % device, file=sys.stderr)
    model = model.to(device)

    #优化器选择
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #参数设置
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []#历史模型得分
    train_time = begin_time = time.time()

    print("开始训练")
    for e in range(1, EPOCH+1):
        
        for ind in range(0, len(train_q), BATCH_SIZE):
            train_iter += 1#计算BATCH次数

            optimizer.zero_grad()

            batch_q = train_q[ind:ind+BATCH_SIZE]
            batch_a = train_a[ind:ind+BATCH_SIZE]

            #计算loss
            losses = -model(batch_q, batch_a)
            batch_loss = losses.sum()
            loss = batch_loss / BATCH_SIZE
            loss.backward()
            #梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            #展示参数设置
            batch_loss_val = batch_loss.item()
            report_loss += batch_loss_val
            cum_loss += batch_loss_val

            tgt_words_num_to_predict =sum(len(s[1:]) for s in train_a) #计算句子长度,TODO:可能需要添加<s>
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += BATCH_SIZE
            cum_examples += BATCH_SIZE
            
            #设定每十次batch展示一次当前参数值
            if train_iter % 10 == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.
            
            #对模型进行评价，当经过800个batch之后对模型进行一次#TODO:修改大小
            if train_iter % 800 == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)                
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                #数据载入
                dev_q = utils.load_data("./data/q_dev.npy")
                dev_a = utils.load_data("./data/a_dev.npy")
                print("开始评测", file=sys.stderr)
                if valid_num == 2:
                    print(dev_q)
                dev_ppl = evaluate_ppl(model, dev_q, dev_a, batch_size=10)
                valid_metric = -dev_ppl
                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                if is_better:
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
train()