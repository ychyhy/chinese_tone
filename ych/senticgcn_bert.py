# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ych import *
from torch.nn import CrossEntropyLoss, MSELoss

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, text, adj): #[16, 85, 768]  [16, 85, 85]
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight) #[16, 85, 768] 很大 超过1
        denom = torch.sum(adj, dim=2, keepdim=True) + 1 #[16, 85, 1]
        output = torch.matmul(adj, hidden) / denom #[16, 85, 768]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SenticGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SenticGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        #self.dropout = nn.Dropout(opt.dropout)
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        #self.text_lstm = DynamicLSTM(768, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        #self.gc4 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc5 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc6 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc7 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        #self.gc8 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim) #768 3
        self.text_embed_dropout = nn.Dropout(0.3)
    # text_out, aspect_double_idx: aspect的位置[start,end], text_len, aspect_len
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len): #词级别的相对位置编码
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        #batch_size = len(x)
        #seq_len = len(x[1])
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len) #离aspect越远 权重越小
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1,self.opt.max_seq_len)):
                weight[i].append(0) #aspect = 0
            for j in range(aspect_double_idx[i,1]+1, text_len[i]): #右侧
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x #[16, 85, 1] * [16, 85, 768] = [16, 85, 768]

    def mask(self, x, aspect_double_idx): #[16, 85, 768] [16, 2]
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, self.opt.max_seq_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i,1]+1, self.opt.max_seq_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device) #[16, 85, 1]
        return mask*x #[16, 85, 768]

    def forward(self, inputs):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids, left_indices, adj = inputs # [16,85]           [16,85]       [16,85]           [16,85]         [16,85]     [16,85,85]
        text_len = torch.sum(text_indices != 0, dim=-1) 
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        #text = self.embed(text_indices)
        #text = self.text_embed_dropout(text)
        #text_out, (_, _) = self.text_lstm(text, text_len)
        # [16,85,768]
        encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)

        text_out = encoder_layer

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc1(text_out, adj))
        #x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc5(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc6(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc7(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc8(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        # x: [16, 85, 768] 过gcn的token表示
        x = self.mask(x, aspect_double_idx) #aspect表示 [16, 85, 768]
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2)) #[16, 85, 85] token相似度
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)  #[16, 1, 85] token 对 整句话 的 相似度(重要性)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim 相似度*value  [16, 768] 整句表示
        output = self.fc(x)
        return output

class ClauseBert_GCN(nn.Module):
    # ClauseBert token 输入 GCN（三种图：依存图、情感图、情感增强的依存图）然后pooling
    # 输入去除[c],但包含[ch-0]的sample
    def __init__(self, bertModel):
        super(ClauseBert_GCN, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        bert_dim = 768
        self.gc1 = GraphConvolution(bert_dim, bert_dim)
        self.gc2 = GraphConvolution(bert_dim, bert_dim)
        self.gc3 = GraphConvolution(bert_dim, bert_dim)
        self.fc = nn.Linear(bert_dim, 16)
        
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

class Bert_AvgPooling_GCN(nn.Module):
    # ClauseBert token 输入 GCN（三种图：依存图、情感图、情感增强的依存图）然后pooling
    # 输入去除[c],但包含[ch-0]的sample
    def __init__(self, bertModel):
        super(Bert_AvgPooling_GCN, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        bert_dim = 768
        self.gc1 = GraphConvolution(bert_dim, bert_dim)
        self.gc2 = GraphConvolution(bert_dim, bert_dim)
        self.gc3 = GraphConvolution(bert_dim, bert_dim)
        self.fc = nn.Linear(768, 16)
    
    def pad_word2sentence(self,sentence_output,max_words):
        return torch.cat((sentence_output, torch.randn([max_words-sentence_output.shape[0],768],dtype=torch.float).to('cuda')), dim=0)

        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, clause_L=None, 
                words_L=None, words_num_in_clause=None, clause_position=None, 
                words_num_in_sentence=None, adj=None, loss_type='ce'):
        """
        clause_L: the length of clause
        words_L:  the length of word
        words_num_in_clause: the num of words in clause
        words_num_in_sentence: the num of words in sentence
        adj: padding graph matrix
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
        # clause_position_mask 获取token表示
        clause_position_mask = clause_position.unsqueeze(-1).expand(sequence_output.size()) # 如何获取子句表示
        clause_token_output = torch.masked_select(sequence_output, (clause_position_mask == 1)).view(-1,768)
        # clause_token_output = self.dropout(clause_token_output) #533 768
        words_tuple = torch.split(clause_token_output, words_L, dim = 0) 
        # 平均池化
        # clause_output = torch.stack([torch.mean(tmp,dim=0) for tmp in clause_tuple]) 
        # 最大池化
        word_output = torch.stack([torch.max(tmp,dim=0).values for tmp in words_tuple])
        # word_output = torch.stack([torch.mean(tmp,dim=0) for tmp in words_tuple])

        sentence_tuple = torch.split(word_output, words_num_in_sentence, dim = 0)
        # 将词序列 padding到最大长度
        max_words = max(words_num_in_sentence)
        word2sentence_output = torch.stack([self.pad_word2sentence(tmp,max_words) for tmp in sentence_tuple]).reshape(-1,max_words,768) #[11, 45, 768]
        x = F.relu(self.gc1(word2sentence_output, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))

        # x: [16, 85, 768] 过gcn的token表示
        words_sentence_output = torch.cat([x[i,:l,:] for i,l in enumerate(words_num_in_sentence)],dim=0)
        words_clause_tuple = torch.split(words_sentence_output, words_num_in_clause, dim = 0)
        words_clause_output = torch.stack([torch.max(tmp,dim=0).values for tmp in words_clause_tuple])
        logits = self.fc(words_clause_output) # x,16

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

class Bert_AvgPooling_GCN_c(nn.Module):
    # ClauseBert token 输入 GCN（三种图：依存图、情感图、情感增强的依存图）然后pooling
    # 输入去除[c],但包含[ch-0]的sample
    def __init__(self, bertModel):
        super(Bert_AvgPooling_GCN_c, self).__init__()
        self.num_labels = 16
        self.bert = bertModel
        self.dropout = nn.Dropout(0.1)
        bert_dim = 768
        self.gc1 = GraphConvolution(bert_dim, bert_dim)
        self.gc2 = GraphConvolution(bert_dim, bert_dim)
        self.gc3 = GraphConvolution(bert_dim, bert_dim)
        self.gc4 = GraphConvolution(bert_dim, bert_dim)
        self.gc5 = GraphConvolution(bert_dim, bert_dim)
        self.fc = nn.Linear(768*2, 16)
        self.fc1 = nn.Linear(768, 16)
    
    def pad_word2sentence(self,sentence_output,max_words):
        return torch.cat((sentence_output, torch.randn([max_words-sentence_output.shape[0],768],dtype=torch.float).to('cuda')), dim=0)

        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, output_hidden_states=None, clause_L=None, 
                words_L=None, words_num_in_clause=None, clause_position=None, 
                words_num_in_sentence=None, adj=None, loss_type='ce',
                c_position=None):
        """
        clause_L: the length of clause
        words_L:  the length of word
        words_num_in_clause: the num of words in clause
        words_num_in_sentence: the num of words in sentence
        adj: padding graph matrix
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
        # clause_position_mask 获取token表示
        clause_position_mask = clause_position.unsqueeze(-1).expand(sequence_output.size())
        c_position_mask = c_position.unsqueeze(-1).expand(sequence_output.size())
        clause_token_output = torch.masked_select(sequence_output, (clause_position_mask == 1)).view(-1,768)
        c_output = torch.masked_select(sequence_output, (c_position_mask == 1)).view(-1,768)

        # clause_token_output = self.dropout(clause_token_output) #533 768
        words_tuple = torch.split(clause_token_output, words_L, dim = 0) 
        # 平均池化
        # clause_output = torch.stack([torch.mean(tmp,dim=0) for tmp in clause_tuple]) 
        # 最大池化
        word_output = torch.stack([torch.max(tmp,dim=0).values for tmp in words_tuple])
        # word_output = torch.stack([torch.mean(tmp,dim=0) for tmp in words_tuple])

        sentence_tuple = torch.split(word_output, words_num_in_sentence, dim = 0)
        # 将词序列 padding到最大长度
        max_words = max(words_num_in_sentence)
        word2sentence_output = torch.stack([self.pad_word2sentence(tmp,max_words) for tmp in sentence_tuple]).reshape(-1,max_words,768) #[11, 45, 768]
        x = F.relu(self.gc1(word2sentence_output, adj))
        # x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, adj))
        # x = F.relu(self.gc4(x, adj))
        # x = F.relu(self.gc5(x, adj))

        # x: [16, 85, 768] 过gcn的token表示
        words_sentence_output = torch.cat([x[i,:l,:] for i,l in enumerate(words_num_in_sentence)],dim=0)
        words_clause_tuple = torch.split(words_sentence_output, words_num_in_clause, dim = 0)
        words_clause_output = torch.stack([torch.max(tmp,dim=0).values for tmp in words_clause_tuple])
        clause_output = torch.cat([c_output,words_clause_output],dim=1)
        logits = self.fc(clause_output) # x,16

        # logits = self.fc1(c_output) # x,16

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