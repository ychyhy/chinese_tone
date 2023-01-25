# -*- coding: utf-8 -*-

import numpy as np
#import spacy
import pickle
from tqdm import tqdm, trange
import pandas as pd

#nlp = spacy.load('en_core_web_sm')
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) #for import
os.chdir(BASE_DIR) #for relative path

def load_sentic_word():
    """
    load senticNet
    """
    path = './DIT_chinese_sentiment_dictionary.txt'
    senticNet = {}
    fp = open(path, 'r', encoding='utf-8-sig')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(document, senticNet):
    seq_len = len(document)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
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
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    return matrix

def process(filename,dep_graph):
    senticNet = load_sentic_word()
    df = pd.read_csv(filename)
    idx2graph = {}
    fout = open(filename+'.sentic', 'wb')
    for i,row in tqdm(df.iterrows(),total=len(df)):
        document = dep_graph[i][1]
        adj_matrix = dependency_adj_matrix(document, senticNet)
        idx2graph[i] = (adj_matrix,document)
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
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

