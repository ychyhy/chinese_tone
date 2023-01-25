# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import os,sys

from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) #for import
os.chdir(BASE_DIR) #for relative path

from ltp import LTP
import torch
ltp = LTP("LTP/base")  # default load small model
if torch.cuda.is_available():
    # ltp.cuda()
    ltp.to("cuda")

def dependency_adj_matrix(text):
    document = ltp.pipeline([text], tasks=["cws", "dep"])
    dep = (np.array(document.dep[0]['head'])-1).tolist() #Dependency relation results 
    seq_len = len(document.cws[0])
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for i,token in enumerate(document.cws[0]):
        # construct weights
        matrix[i][i] = 1
        if dep[i] > 0: #remove root
            matrix[i][dep[i]] = 1
            matrix[dep[i]][i] = 1
    return matrix,document.cws[0]

def process(filename):
    df = pd.read_csv(filename)
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i,row in tqdm(df.iterrows(),total=len(df)):
        adj_matrix,document = dependency_adj_matrix(row.pre_text)
        idx2graph[i] = (adj_matrix,document)
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    # use the participle from ltp
    # f = open('./datasets/tone/preprocess_train.csv'+'.graph', 'rb')
    # train_dep_graph = pickle.load(f)
    # f = open('./datasets/tone/preprocess_test.csv'+'.graph', 'rb')
    # test_dep_graph = pickle.load(f)
    process('../data/sentc_train.csv')
    process('../data/sentc_dev.csv')
    process('../data/sentc_test.csv')
