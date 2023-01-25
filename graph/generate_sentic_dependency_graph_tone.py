# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import spacy
import pickle
import os,sys

from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) #for import
os.chdir(BASE_DIR) #for relative path
nlp = spacy.load('en_core_web_sm')
from ltp import LTP
import torch
ltp = LTP("LTP/base")  
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

def load_sentic_word():
    """
    load senticNet
    """
    path = './DIT_chinese_sentiment_dictionary.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(document, senticNet, text):

    output = ltp.pipeline([text], tasks=["cws", "dep"])
    dep = (np.array(output.dep[0]['head'])-1).tolist() 
    assert output.cws[0] == document
    seq_len = len(document)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    sent_matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for i in range(seq_len):
        wordi = document[i]
        for j in range(seq_len):
            wordj = document[j]
            if wordi in senticNet and wordj in senticNet:
                # sentic = float(senticNet[word]) + 1.0
                sentic = abs(float(senticNet[wordi])+float(senticNet[wordj]))
            elif wordi in senticNet:
                sentic = abs(float(senticNet[wordi]))
            elif wordj in senticNet:
                sentic = abs(float(senticNet[wordj]))
            else:
                sentic = 0
            sent_matrix[i][j] = sentic
    assert (sent_matrix == sent_matrix.T).all()

    for i,token in enumerate(document):
        matrix[i][i] = 1
        if dep[i] > 0: 
            matrix[i][dep[i]] = 1
            matrix[dep[i]][i] = 1

    assert (matrix == matrix.T).all()
    result = matrix*sent_matrix

    for i,token in enumerate(document):
        if result[i][i] == 0:
            result[i][i] = 1
    
    return result

def process(filename,dep_graph):
    senticNet = load_sentic_word()
    df = pd.read_csv(filename)
    idx2graph = {}
    fout = open(filename+'.graph_sdat', 'wb')
    for i,row in tqdm(df.iterrows(),total=len(df)):
        document = dep_graph[i][1]
        adj_matrix = dependency_adj_matrix(document, senticNet, row.pre_text)
        idx2graph[i] = (adj_matrix,document)
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    f = open('../data/sentc_train.csv'+'.graph', 'rb')
    train_dep_graph = pickle.load(f)
    f = open('../data/sentc_dev.csv'+'.graph', 'rb')
    dev_dep_graph = pickle.load(f)
    f = open('../data/sentc_test.csv'+'.graph', 'rb')
    test_dep_graph = pickle.load(f)
    process('../data/sentc_train.csv',train_dep_graph)
    process('../data/sentc_dev.csv',dev_dep_graph)
    process('../data/sentc_test.csv',test_dep_graph)
    
