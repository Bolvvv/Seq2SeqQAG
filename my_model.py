import torch
import torch.nn as nn
from utils import pad_batch_word_to_id
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class my_model(nn.Module):
    
    def __init__(self, q_map, a_map, emb_size, hidden_size, padding_idx):
        super(my_model, self).__init__()
        self.q_emb = nn.Embedding(len(q_map), emb_size, padding_idx=q_map['<pad>'])#设置词嵌入，并设置padding为0
        self.a_emb = nn.Embedding(len(a_map), emb_size, padding_idx=a_map['<pad>'])
        self.encode = nn.LSTM(emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.decode = nn.LSTMCell(hidden_size+emb_size, hidden_size)#这一步是将目标词汇和上一步的decode输出词汇进行链接
        self.doubleH2oneH = nn.Linear(2*hidden_size, hidden_size)#将双向LSTM的输出值h变换为单hidden
        self.doubleC2oneC = nn.Linear(2*hidden_size, hidden_size)#将双向LSTM的输出值c变换为单hidden
        self.att_projection = nn.Linear(2*hidden_size, hidden_size)#将encode和decode的h输出组合生成attention
        self.softmax = nn.Softmax()
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size)#将从注意力生成的encode输出与当前decode的输出组合在一起生成新的hidden
        self.target_answer_projection = nn.Linear(hidden_size, len(a_map))#将decode的输出hidden转换为answer_map的大小，再使用softmax求出需要用到的词

    def forward(self, question2list_id, answer2list_id):
        q_pad, a_pad, q_len, a_len = pad_batch_word_to_id(question2list_id, answer2list_id, self.pad_token, self.q_map, self.a_map, self.device)
        out_put, decode_init_state = self.forward_step(q_pad, q_len)#准备好解码需要用到的值。其中out_put还需要经过attention层，decode_init_state包含了hidden和cell
        


    def forward_step(self, q_pad, q_len):
        X = self.q_emb(q_pad)#在此步进行词嵌入，X的形式为(batch_size, question_len, emb_size)
        #参考:https://blog.csdn.net/lssc4205/article/details/79474735
        X = pack_padded_sequence(X, q_len, batch_first=True)#对嵌入后的矩阵进行压缩，告诉LSTM有些pad值是不需要的
        out_put, (h_n, c_n) = self.encode(X)#dim(h_n) = dim(c_n) = (2, batch_size, hidden_size)
        out_put_padded, _ = pad_packed_sequence(out_put, batch_first=True)#恢复回来out_put:(batch_size, question_len, num_directions * hidden_size)
        init_decoder_hidden = self.doubleH2oneH(torch.cat((h_n[0], h_n[1]), dim=1))#易知dim(h_n)=(2, b, h),则dim(h_n[0])=(b,h),设置dim=1则表明在h纬度上拼接
        init_decoder_cell = self.doubleC2oneC(torch.cat((c_n[0], c_n[1]), dim=1))
        
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        return out_put_padded, dec_init_state


        
