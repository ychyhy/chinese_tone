import cmath
from math import gamma
from turtle import forward
from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from zmq import device
from .pairscl_util import masked_softmax, weighted_sum, sort_by_seq_lens
from ych import *
from .CRF import CRF

class Output():
    ...

class BertForCL(nn.Module):
    def __init__(self, bertModel):
        super(BertForCL, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 16)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                labels=None,loss_type = 'focal'):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        pooled_output = outputs['hidden_states'][-1][:,0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output
        
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, encoder, num_classes=3):
        super(LinearClassifier, self).__init__()
        dim_mlp = encoder.fc.weight.shape[1]
        #768  3
        self.fc = nn.Linear(dim_mlp, num_classes)
    
    def forward(self, features):
        return self.fc(features)

class PairSupConBert(nn.Module):
    def __init__(self, encoder, dropout=0.5, is_train=True):
        super(PairSupConBert, self).__init__()
        self.encoder = encoder.bert
        self.config = encoder.config
        self.dim_mlp = encoder.fc.weight.shape[1]
        self.dropout = dropout
        self.is_train = is_train
        self.attention = SoftmaxAttention()
        self.projection = nn.Sequential(
            nn.Linear(4*self.dim_mlp, self.dim_mlp),
            nn.ReLU())
        self.pooler = nn.Sequential(nn.Linear(4*self.dim_mlp,self.dim_mlp),
                                    encoder.bert.pooler)
        self.head = nn.Sequential(nn.Linear(self.dim_mlp,self.dim_mlp),
                                        nn.ReLU(inplace=True),
                                        encoder.fc)
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, position_ids=None, head_mask=None, inputs_embeds=None):
        input_ids2 = input_ids * token_type_ids
        input_ids1 = input_ids - input_ids2
        feat1 = self.encoder(input_ids1, #['last_hidden_state', 'pooler_output'])
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds)
        feat2 = self.encoder(input_ids2,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds)
        encoded_premises = feat1[0] #64, 128, 768
        encoded_hypotheses = feat2[0]
        attended_premises, attended_hypotheses = self.attention(encoded_premises, attention_mask, encoded_hypotheses, attention_mask) #64, 128, 768
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises], dim=-1) #64, 128, 3072
        enhanced_hypotheses = torch.cat([encoded_hypotheses, attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],dim=-1)
        projected_premises = self.projection(enhanced_premises) #64, 128, 768
        projected_hypotheses = self.projection(enhanced_hypotheses)
        pair_embeds = torch.cat([projected_premises, projected_hypotheses, projected_premises - projected_hypotheses, projected_premises * projected_hypotheses], dim=-1)
        pair_output = self.pooler(pair_embeds) # bs,768
        feat = F.normalize(self.head(pair_output), dim=1) #bs,num_label
        outputs = Output()
        outputs.logits = feat
        outputs.pooled_output = pair_output
        return outputs # bs,labels
            
class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """

    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.

        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                                              .contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                                        .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.

    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase),\
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.

        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx =\
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class ClauseBert(nn.Module):
    # bert-[c] 需要更换数据
    def __init__(self, bertModel):
        super(ClauseBert, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 16)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='ce'):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        c_idx_mask = c_idx.unsqueeze(-1).expand(sequence_output.size()) #
        clause_output = torch.masked_select(sequence_output, (c_idx_mask == 1)).view(-1,768)
        clause_output = self.dropout(clause_output) 
        logits = self.fc(clause_output) # x,16

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class Bert_AvgPooling(nn.Module):
    def __init__(self, bertModel):
        super(Bert_AvgPooling, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 16)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, clause_L=None, clause_position=None, loss_type='ce'):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        clause_position_mask = clause_position.unsqueeze(-1).expand(sequence_output.size()) # 如何获取子句表示
        clause_token_output = torch.masked_select(sequence_output, (clause_position_mask == 1)).view(-1,768)
        clause_token_output = self.dropout(clause_token_output) #533 768
        clause_tuple = torch.split(clause_token_output, clause_L, dim = 0)
        # 平均池化
        clause_output = torch.stack([torch.mean(tmp,dim=0) for tmp in clause_tuple]) 
        # 最大池化
        # clause_output = torch.stack([torch.max(tmp,dim=0).values for tmp in clause_tuple]) 
        logits = self.fc(clause_output) # x,16

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class Bert_LSTM(nn.Module):
    # 单句过bert
    def __init__(self, bertModel,LSTM_config):
        super(Bert_LSTM, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = nn.LSTM(input_size=768, hidden_size=self.rnn_hidden,
                        bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
                        num_layers=LSTM_config['num_layers'],
                        dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0)
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
    
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='focal',index=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )

        '''
        向量化编程，将数据操作处理好封装成类，然后传进来。这样可以避免正向传播中创建tensor，保证正向过程与数据处理过程分离
        '''
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        
        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1


        # 修改
        lstm_input = self.dropout(empty_lstm_input)
        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, (h1_0, c1_0)) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len, h0=None, c0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None: 
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, c0))
        else: 
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)
            return out, (ht, ct)

class Bert_DynamicLSTM(nn.Module):
    # 单句过bert 然后bilstm
    def __init__(self, bertModel,LSTM_config):
        super(Bert_DynamicLSTM, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
    
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='focal',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs['hidden_states'][-1] #64,118,768
        clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls 118,768

        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class Bert_DynamicLSTM_CRF(nn.Module):
    # 单句过bert 然后bilstm
    def __init__(self, bertModel,LSTM_config):
        super(Bert_DynamicLSTM_CRF, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
        self.CRF_layer = CRF(self.num_labels, LSTM_config['batch_first'])

    
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='focal',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs['hidden_states'][-1] #64,118,768
        clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls 118,768

        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]
        loss = self.CRF_layer(logits.unsqueeze(0), labels.unsqueeze(0), None, 'sum')
        return (loss,logits,labels) if loss is not None else logits

class Bert_AvgPooling_DynamicLSTM(nn.Module):
    def __init__(self, bertModel, LSTM_config):
        super(Bert_AvgPooling_DynamicLSTM, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
        # self.fc = nn.Linear(768, 16)
        
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, clause_L=None, clause_position=None, loss_type='ce',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        clause_position_mask = clause_position.unsqueeze(-1).expand(sequence_output.size()) # 如何获取子句表示
        clause_token_output = torch.masked_select(sequence_output, (clause_position_mask == 1)).view(-1,768)
        clause_token_output = self.dropout(clause_token_output) #533 768
        clause_tuple = torch.split(clause_token_output, clause_L, dim = 0)
        # 平均池化
        # clause_output = torch.stack([torch.mean(tmp,dim=0) for tmp in clause_tuple]) #TODO 如何避免循环
        # 最大池化
        clause_output = torch.stack([torch.max(tmp,dim=0).values for tmp in clause_tuple]) #TODO 如何避免循环
        # LSTM
        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        # clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]
        
        # logits = self.fc(clause_output) # x,16

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class Bert_AvgPooling_DynamicLSTM_CRF(nn.Module):
    def __init__(self, bertModel, LSTM_config):
        super(Bert_AvgPooling_DynamicLSTM_CRF, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
        self.CRF_layer = CRF(self.num_labels, LSTM_config['batch_first'])
        # self.fc = nn.Linear(768, 16)
        
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, clause_L=None, clause_position=None, loss_type='ce',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        clause_position_mask = clause_position.unsqueeze(-1).expand(sequence_output.size()) # 如何获取子句表示
        clause_token_output = torch.masked_select(sequence_output, (clause_position_mask == 1)).view(-1,768)
        clause_token_output = self.dropout(clause_token_output) #533 768
        clause_tuple = torch.split(clause_token_output, clause_L, dim = 0)
        # 平均池化
        # clause_output = torch.stack([torch.mean(tmp,dim=0) for tmp in clause_tuple]) #TODO 如何避免循环
        # 最大池化
        clause_output = torch.stack([torch.max(tmp,dim=0).values for tmp in clause_tuple]) #TODO 如何避免循环
        # LSTM
        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        # clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]
        loss = self.CRF_layer(logits.unsqueeze(0), labels.unsqueeze(0), None, 'sum')
        return (loss,logits,labels) if loss is not None else logits

class ClauseBert_DynamicLSTM(nn.Module):
    # bert-[c] 需要更换数据
    def __init__(self, bertModel, LSTM_config):
        super(ClauseBert_DynamicLSTM, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
        # self.fc = nn.Linear(768, 16)
        
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='ce',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        c_idx_mask = c_idx.unsqueeze(-1).expand(sequence_output.size()) #
        clause_output = torch.masked_select(sequence_output, (c_idx_mask == 1)).view(-1,768)
        clause_output = self.dropout(clause_output)
        # LSTM
        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        # clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif loss_type == 'ce':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif loss_type=='focal': #focal loss
                loss_fct = Focal_loss(
                    alpha=[0.2,0.5,0.5,0.5,0.5,0.8,1,0.5,0.8,0.8,0.8,1,0.8,0.5,0.5,0.5],
                    gamma=2,
                    num_classes = 16
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                raise Exception(print('no loss type'))
        output = logits
        return (loss,output,labels) if loss is not None else output

class ClauseBert_DynamicLSTM_CRF(nn.Module):
    # bert-[c] 需要更换数据
    def __init__(self, bertModel, LSTM_config):
        super(ClauseBert_DynamicLSTM_CRF, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        self.rnn_hidden = LSTM_config['hidden_size'] // 2 if LSTM_config['bidirectional'] else LSTM_config['hidden_size']
        self.num_layers = LSTM_config['num_layers']
        self.num_directions = 2 if LSTM_config['bidirectional'] else 1
        self.BiLSTM = DynamicLSTM(
            input_size=768, hidden_size=self.rnn_hidden,
            bidirectional=LSTM_config['bidirectional'], batch_first=LSTM_config['batch_first'],
            num_layers=LSTM_config['num_layers'],
            dropout=LSTM_config['dropout'] if LSTM_config['num_layers'] > 1 else 0,
            rnn_type = LSTM_config['rnn_type']
        )
        self.hidden2tag = nn.Linear(2*self.rnn_hidden if LSTM_config['bidirectional'] else self.rnn_hidden, self.num_labels) 
        self.CRF_layer = CRF(self.num_labels, LSTM_config['batch_first'])
        # self.fc = nn.Linear(768, 16)
        
    def init_hidden(self, batch_size):
        # torch.randn 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。返回一个张量
        return (torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'),
                torch.randn(self.num_layers * self.num_directions, batch_size, self.rnn_hidden).to(device='cuda'))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, c_idx=None, loss_type='ce',index=None, lens = None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
        
        sequence_output = outputs['hidden_states'][-1] #64,118,768
        c_idx_mask = c_idx.unsqueeze(-1).expand(sequence_output.size()) #
        clause_output = torch.masked_select(sequence_output, (c_idx_mask == 1)).view(-1,768)
        clause_output = self.dropout(clause_output)
        # LSTM
        batch_size = index[0][-1].item()+1
        squence_len = torch.max(torch.bincount(index[0])).item()
        # clause_output = torch.split(sequence_output,1,dim=1)[0].squeeze(1) #cls
        empty_lstm_input = torch.randn(768).expand(batch_size,squence_len,-1).to('cuda')
        empty_lstm_input[index] = clause_output
        empty_mask = torch.zeros(batch_size,squence_len).to('cuda')
        empty_mask[index] = 1
        lstm_input = self.dropout(empty_lstm_input)

        h1_0, c1_0 = self.init_hidden(batch_size)
        BiLSTM_output, _ = self.BiLSTM(lstm_input, lens, h1_0, c1_0) #bs,sentence,300
        empty_mask = empty_mask.unsqueeze(-1).expand(BiLSTM_output.size())
        BiLSTM_output = torch.masked_select(BiLSTM_output,(empty_mask == 1)).view(-1,300)
        logits = self.hidden2tag(BiLSTM_output)  # [B * L * tag_num]
        loss = self.CRF_layer(logits.unsqueeze(0), labels.unsqueeze(0), None, 'sum')
        return (loss,logits,labels) if loss is not None else logits