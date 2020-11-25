import torch
import torch.nn as nn
from utils import pad_batch_word_to_id
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
import sys

class my_model(nn.Module):
    
    def __init__(self, q_map, a_map, emb_size, hidden_size, dropout_rate=0.2):
        super(my_model, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.q_map = q_map
        self.a_map = a_map
        self.q_emb = nn.Embedding(len(q_map), emb_size, padding_idx=q_map['<pad>'])#设置词嵌入，并设置padding为0
        self.a_emb = nn.Embedding(len(a_map), emb_size, padding_idx=a_map['<pad>'])
        self.encode = nn.LSTM(emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.decode = nn.LSTMCell(hidden_size+emb_size, hidden_size)#这一步是将目标词汇和上一步的decode输出词汇进行链接
        self.doubleH2oneH = nn.Linear(2*hidden_size, hidden_size)#将双向LSTM的输出值h变换为单hidden
        self.doubleC2oneC = nn.Linear(2*hidden_size, hidden_size)#将双向LSTM的输出值c变换为单hidden
        self.att_projection = nn.Linear(2*hidden_size, hidden_size)#将encode和decode的h输出组合生成attention
        self.combined_output_projection = nn.Linear(3*hidden_size, hidden_size)#将从注意力生成的encode输出与当前decode的输出组合在一起生成新的hidden
        self.target_answer_projection = nn.Linear(hidden_size, len(a_map))#将decode的输出hidden转换为answer_map的大小，再使用softmax求出需要用到的词
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, question2list_id, answer2list_id):
        q_pad, q_len = pad_batch_word_to_id(question2list_id ,self.q_map, self.device)
        a_pad, a_len = pad_batch_word_to_id(answer2list_id, self.a_map, self.device)
        out_put, decode_init_state = self.forward_step_encode(q_pad, q_len)#准备好解码需要用到的值。其中out_put还需要经过attention层，decode_init_state包含了hidden和cell
        enc_masks = self.generate_sent_masks(out_put, q_len)
        combined_outputs = self.forward_step_decode(out_put, enc_masks, decode_init_state, a_pad)
        P = F.log_softmax(self.target_answer_projection(combined_outputs), dim=-1)#(tgt_len, b, map_len)
        #为遵照源代码，在此步将a_pad的纬度由[b,tgt_len]转换为[tgt_len, b]
        a_pad = a_pad.permute(1,0)
        target_masks = (a_pad != self.a_map['<pad>']).float()
        target_gold_words_log_prob = torch.gather(P, index=a_pad[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def forward_step_encode(self, q_pad, q_len):
        X = self.q_emb(q_pad)#在此步进行词嵌入，X的形式为(batch_size, question_len, emb_size)
        #参考:https://blog.csdn.net/lssc4205/article/details/79474735
        X = pack_padded_sequence(X, q_len, batch_first=True)#对嵌入后的矩阵进行压缩，告诉LSTM有些pad值是不需要的
        out_put, (h_n, c_n) = self.encode(X)#dim(h_n) = dim(c_n) = (2, batch_size, hidden_size)
        out_put_padded, _ = pad_packed_sequence(out_put, batch_first=True)#恢复回来out_put:(batch_size, question_len, num_directions * hidden_size)
        init_decoder_hidden = self.doubleH2oneH(torch.cat((h_n[0], h_n[1]), dim=1))#易知dim(h_n)=(2, b, h),则dim(h_n[0])=(b,h),设置dim=1则表明在h纬度上拼接
        init_decoder_cell = self.doubleC2oneC(torch.cat((c_n[0], c_n[1]), dim=1))
        
        dec_init_state = (init_decoder_hidden, init_decoder_cell)#((b, h), (b, h))
        return out_put_padded, dec_init_state
    
    def forward_step_decode(self, out_put, encode_mask, decode_init_state, a_pad):
        """
        在此步进行解码操作
        """
        # a_pad = a_pad.permute(1,0)
        # a_pad = a_pad[:-1]#target需要添加<END>标签并去除
        # a_pad = a_pad.permute(1,0)
        dec_state = decode_init_state#初始化decode_state,((b, h), (b, h))

        batch_size = out_put.size(0)#初始化batch
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        #初始化一个list用于收集每次的输出o
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(out_put)#将encode层的输出进行转换，(b, src_len, 2*h)->(b, src_len, h)
        Y = self.a_emb(a_pad)#(b, tgt_len, e)
        Y = Y.permute(1,0,2)#(tgt_len, b, e)
        for Y_t in torch.split(Y, 1):
            Y_t = torch.squeeze(Y_t, dim=0)#去除掉(1, b, e)中的1
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)#对输入值进行拼接,(b, e+h)
            dec_state, o_t, e_t = self.decode_step(Ybar_t, dec_state, out_put, enc_hiddens_proj, encode_mask)
            o_prev = o_t
            combined_outputs.append(o_t)
        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def decode_step(self, Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, encode_mask):

        dec_state = self.decode(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        #使用torch.unsqueeze将dec_hidden的(b,h)转换为(b, h, 1),再使用bmm进行矩阵乘法
        e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2))#(b, src_len, 1)
        e_t = torch.squeeze(e_t, dim=2)#(b, src_len)

        #由于求得的attention scores存在pad，因此需要使用mask对其进行遮盖
        if encode_mask is not None:
            e_t.data.masked_fill_(encode_mask.byte(), -float('inf'))
        
        alpha_t = torch.unsqueeze(F.softmax(e_t, dim=1), dim=1)
        a_t = torch.squeeze(torch.bmm(alpha_t, enc_hiddens), dim=1)#(b, 2*h)#TODO:没想清楚为什么在这一层还需要结合output
        u_t = torch.cat((a_t, dec_hidden), dim=1)#(b, 3*h)
        v_t = self.combined_output_projection(u_t) #(b, h)
        O_t = self.dropout(torch.tanh(v_t))
        return dec_state, O_t, e_t
    
    def generate_sent_masks(self, enc_hiddens, source_lengths):
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @property
    def device(self):
        return self.q_emb.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(q_map=params['q_map'], a_map=params['a_map'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(emb_size=self.emb_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'q_map': self.q_map,
            'a_map': self.a_map,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
    
    def beam_search(self, src_sent, beam_size=5, max_decoding_time_step=70):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = pad_batch_word_to_id([src_sent], self.q_map, self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses