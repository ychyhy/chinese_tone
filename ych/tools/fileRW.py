from email.policy import strict
import os
import json
import sys
import pickle
# import ujson

def readJson(path):
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict

# ujson 会识别json中的错误
def writeJson(path,new_dict):
    with open(path,"w") as f:
        json.dump(new_dict,f)

def readPickle(path):
    with open(path,'rb') as load_f:
        load_dict = pickle.load(load_f)
    return load_dict

def writePickle(path,new_dict):
    with open(path,"wb") as f:
        pickle.dump(new_dict,f)

#datatime的使用

