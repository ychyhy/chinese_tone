import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def hidden_states_to_pooled_output(hidden_states):
    '''BertForSequenceClassification does not return pooled_output'''
    pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
    # pooled_output = torch.cat(tuple([hidden_states[i] for i in [-1]]), dim=-1)
    pooled_output = pooled_output[:, 0, :]
    pooled_output = F.dropout(pooled_output)
    pooled_output = F.normalize(pooled_output, dim=1)
    # classifier of course has to be 4 * hidden_dim, because we concat 4 layers
    # logits = self.classifier(pooled_output)
    return pooled_output

