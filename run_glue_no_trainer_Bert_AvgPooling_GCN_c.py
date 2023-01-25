# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import pickle
import random
from pathlib import Path
import string
from tkinter import _flatten
import datasets
import numpy as np
import pandas as pd
import torch
import transformers 
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_metric
from huggingface_hub import Repository
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding, PretrainedConfig,
                          SchedulerType, default_data_collator, get_scheduler)
from transformers.utils import get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from ych import *
from torch.utils.tensorboard import SummaryWriter


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true", #command carry:True
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=20220708, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--resume_from_path",
        type=str,
        default=None,
        help="If the training should continue from a save model path.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--output_bad_case",
        action="store_true",
        help="Whether or not to output bad case.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default='dep',
        help="dep, sentic, dep_sentic",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def get_c_idx(batch):
    c_batch = []
    c_idx = np.nonzero(batch['input_ids'].cpu().numpy() == 9) #[c]
    tmp = c_idx[0][0]
    l = []
    for i in range(len(c_idx[0])):
        if c_idx[0][i] == tmp:
            l.append(c_idx[1][i])
        else:
            c_batch.append(l)
            tmp = c_idx[0][i]
            l = [c_idx[1][i]]
    c_batch.append(l)
    return c_batch

def get_c_idx_mask(batch):
    return torch.where(batch['input_ids']==9,1,0)

def get_clause_position(input_ids):
    # 构建子句token mask
    goal = []
    for i in input_ids:
        goal.append(0 if i in [9,102,101,0] else 1)
    return goal

def get_batch_clause_position(batch_input_ids):
    # 构建子句token mask batch 
    goal = []
    for b in batch_input_ids:
        goal.append(get_clause_position(b))
    return goal

def get_clause_L(input_ids):
    # 记录子句长度
    goal = []
    for i in range(0,len(input_ids)):
        if input_ids[i] == 0:
            s = sum(input_ids[0:i])
            if s != 0:
                goal.append(s)
    return [goal[0]] + list(np.diff(goal))

def get_batch_clause_len(batch_input_ids):
    # 记录batch 子句长度
    goal = []
    for b in batch_input_ids:
        goal += get_clause_L(b)
    return goal

def get_labels(raw_dataset):
    return list(map(lambda x : eval(x),raw_dataset['label']))

def get_batch_labels(batch_size,step,train_labels):
    t1 = batch_size
    t2 = len(train_labels)
    end = t2 if (step+1)*t1>t2 else (step+1)*t1
    return train_labels[step*t1:end]

def get_batch_labels_tensor(batch_size,step,train_labels):
    return torch.LongTensor(list(_flatten(get_batch_labels(batch_size,step,train_labels))))

#定义计算信息熵的函数：计算Infor(D) 范围0-logn
def infor(a):
    # a = pd.value_counts(data) / len(data)
    return np.sum(np.log2(a) * a * (-1),axis=-1)

def get_words_num_in_clause(words_L,clause_L):
    assert sum(words_L) == sum(clause_L)
    goal = []
    tmp = 0
    cl = 0
    j = 0
    while cl < len(clause_L):
        while j < len(words_L):
            if sum(words_L[tmp:j+1]) == clause_L[cl]:
                goal.append(j-tmp+1)
                tmp = j+1
                cl += 1
            j += 1
    return goal

def get_words_num_in_sentence(words_num_in_clause,clause_num):
    tmp = 0
    goal = []
    for i in range(len(clause_num)):
        goal.append(sum(words_num_in_clause[tmp:clause_num[i]+tmp]))
        tmp = clause_num[i]+tmp
    assert sum(words_num_in_clause) == sum(goal)
    return goal

def pad_graph(max_seq_len,graphs):
    goal = []
    for g in graphs:
        tmp = np.pad(
            g,
            ((0, max_seq_len-g.shape[0]),(0, max_seq_len-g.shape[0])),
            'constant'
        )
        goal.append(tmp)
    return torch.FloatTensor(goal)

def remove_sample():
    # 针对包含英文符号等无法正确对应LTP和bert分词结果的样本进行去除
    ...


def get_c_position(input_ids):
    goal = []
    for i in input_ids:
        goal.append(1 if i == 9 else 0)
    return goal

def get_batch_c_position(batch_input_ids):
    goal = []
    for b in batch_input_ids:
        goal.append(get_c_position(b))
    return goal

def make_dynamic_batches(dataset, bs, device, graph):
    i = j = 0
    batches = []
    indexs = []
    lens = []
    clause_num = list([len(eval(d['labels'])) for d in dataset]) # 子句数量[8,2,4,5] 2640
    # https://blog.csdn.net/Refrain__WG/article/details/89214660 判断字符
    error = []
    # 可以先使用bert对词语计算长度，然后再pooling
    words_L = [[1 if word.startswith('ch-') or word.isdigit() or word in ['lz','bug','ok','ddos','nba'] else len(word) for word in v[1]] for k,v in graph.items()] #一整句中词长度 用于获取词表示[2,3,2,1,]
    graphs,text = zip(*list(graph.values()))
    while i < len(clause_num):
        if sum(clause_num[i:j+1]) < bs and j != len(clause_num):
            j+=1
        else:
            batch = dataset[i:j]
            ls = clause_num[i:j]
            # pad_sequence
            # input_ids, attention_mask, segment_ids, target
            batch_clause_token_mask = get_batch_clause_position(batch['input_ids'])
            batch_c_token_mask = get_batch_c_position(batch['input_ids'])
            batch['c_position'] = torch.nn.utils.rnn.pad_sequence(\
                list_padding_tensor(batch_c_token_mask),batch_first=True).to(device)
            batch['clause_position'] = torch.nn.utils.rnn.pad_sequence(list_padding_tensor(batch_clause_token_mask),batch_first=True).to(device)
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(list_padding_tensor(batch['input_ids']),batch_first=True).to(device)
            batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(list_padding_tensor(batch['attention_mask']),batch_first=True).to(device)
            batch['token_type_ids'] = torch.nn.utils.rnn.pad_sequence(list_padding_tensor(batch['token_type_ids']),batch_first=True).to(device)
            labels = list(_flatten([eval(tmp) for tmp in batch['labels']]))
            batch['labels'] = torch.LongTensor(labels).to(device)
            cl = get_batch_clause_len(batch_clause_token_mask)
            clause_L = list(_flatten(cl)) #一整句话中子句的长度[4,5,6,]
            batch['words_L'] = list(_flatten(words_L[i:j]))
            # 各个子句中词的数量[3,4,5,]
            for k in range(0,j-i):
                if sum(words_L[i+k]) != sum(batch_clause_token_mask[k]):
                    print(k+i)
                    print(text[k+i])
                    print(sum(words_L[i+k]))
                    print(sum(batch_clause_token_mask[k]))
            batch['words_num_in_clause'] = get_words_num_in_clause(batch['words_L'],clause_L)
            # batch中各整句中词的数量
            words_num_in_sentence = get_words_num_in_sentence(batch['words_num_in_clause'],ls)
            batch['words_num_in_sentence'] = words_num_in_sentence
            batch['adj'] = pad_graph(max(words_num_in_sentence),graphs[i:j]).to(device)
            l = get_batch_lens(ls) #修改数据集后 需要修改
            index = generate_batch_index(ls,device)
            i = j
            batches.append(batch)
            indexs.append(index)
            lens.append(l)
    # return batches,indexs,lens
    return batches

def list_padding_tensor(lists):
    maxL = max(list(map(lambda x:len(x),lists)))
    return torch.LongTensor(list(map(lambda x:x+[0]*(maxL-len(x)),lists)))

def generate_batch_index(clauseL,device):
    clause_id = []
    clause_index = []
    for i in range(len(clauseL)):
        clause_id += [i]*clauseL[i]
        clause_index += list(range(clauseL[i]))
    return (torch.LongTensor(clause_id).to(device),torch.LongTensor(clause_index).to(device))

def get_batch_lens(clauseL):
    return torch.LongTensor(clauseL)

# 参数
LSTM_config = {
  'bidirectional': False,
  'num_layers': 1,
  'hidden_size': 300,
  'dropout': 0.1,
  'input_size': 768,
  'batch_first': True,
  'rnn_type': 'LSTM'
}


def main():
    args = parse_args()
    send_example_telemetry("run_glue_no_trainer", args)
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        label_list = list(range(16))
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
        is_regression = False

    added_token = [
            '[ch-0]',
            '[ch-1]',
            '[ch-2]',
            '[ch-3]',
            '[ch-4]',
            '[ch-5]',
            '[ch-6]',
            '[ch-7]',
            '[c]',
        ]
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, additional_special_tokens=added_token)
    bertModel = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    model = Bert_AvgPooling_GCN_c(
        bertModel
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        sentence1_key, sentence2_key = "text", None

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if isinstance(examples["label"],str):
            result["labels"] = eval(examples["label"])
        else:
            result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=list(set(raw_datasets["train"].column_names)),
            desc="Running tokenizer on dataset",
            load_from_cache_file = True
        )
    logger.info(processed_datasets.column_names)
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    test_dataset = processed_datasets["test"]

    # 读取图
    if args.graph_type == 'dep':
        fin = open('./data/sentc_train.csv.graph', 'rb')
        train_graph = pickle.load(fin)
        fin = open('./data/sentc_dev.csv.graph', 'rb')
        dev_graph = pickle.load(fin)
        fin = open('./data/sentc_test.csv.graph', 'rb')
        test_graph = pickle.load(fin)
        fin.close()
    elif args.graph_type == 'sentic':
        fin = open('./data/sentc_train.csv.sentic', 'rb')
        train_graph = pickle.load(fin)
        fin = open('./data/sentc_dev.csv.sentic', 'rb')
        dev_graph = pickle.load(fin)
        fin = open('./data/sentc_test.csv.sentic', 'rb')
        test_graph = pickle.load(fin)
        fin.close()
    else: #dep_sentic
        fin = open('./data/sentc_train.csv.graph_sdat', 'rb')
        train_graph = pickle.load(fin)
        fin = open('./data/sentc_dev.csv.graph_sdat', 'rb')
        dev_graph = pickle.load(fin)
        fin = open('./data/sentc_test.csv.graph_sdat', 'rb')
        test_graph = pickle.load(fin)
        fin.close()

    bs = args.per_device_train_batch_size
    train_dataloader = make_dynamic_batches(train_dataset, bs, device, train_graph)
    print('finish train_dataloader')
    eval_dataloader = make_dynamic_batches(eval_dataset, bs, device, dev_graph)
    print('finish eval_dataloader')
    test_dataloader = make_dynamic_batches(test_dataset, bs, device, test_graph)
    print('finish test_dataloader')
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    '''
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    '''
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'bert' in n],
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bert' not in n],
            'lr': 1e-2
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
        test_metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric('./arc_metric.py')
        test_metric = load_metric('./ConfusionMatrixMetric.py')

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    best_eval_F1 = 0
    writer = SummaryWriter('./logs')
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            # batch_size, sequence_length, hidden_size
            outputs = model(**batch,output_hidden_states=True,loss_type='ce')
            loss = outputs[0]
            writer.add_scalar("loss",loss,epoch)

            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                # batch_size, sequence_length, hidden_size
                outputs = model(**batch,output_hidden_states=True)
                loss = outputs[0]

            predictions = outputs[1].argmax(dim=-1) if not is_regression else outputs[1].squeeze()
            # predictions, references = accelerator.gather((predictions, outputs[2]))
            predictions, references = accelerator.gather((predictions, batch['labels']))

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")
        writer.add_scalar("micro_f1",eval_metric['micro_f1'],epoch)


        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if max(best_eval_F1,eval_metric['micro_f1']) == eval_metric['micro_f1'] :
            best_eval_F1 = eval_metric['micro_f1']
            if args.output_dir is not None:
                logger.info(f"best_eval_F1: {best_eval_F1}")
                accelerator.wait_for_everyone()
                torch.save(model, args.output_dir+'model.pkl')
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    config.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    writer.close()
    model.eval()
    samples_seen = 0
    all_predictions = []
    all_references = []
    all_confidences = []
    # load best model
    model = torch.load(args.output_dir+'model.pkl')

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            # batch_size, sequence_length, hidden_size
            outputs = model(**batch,output_hidden_states=True)
            loss = outputs[0]

        predictions = outputs[1].argmax(dim=-1) if not is_regression else outputs[1].squeeze()
        confidences = outputs[1].softmax(dim=-1) if not is_regression else outputs[1].squeeze()
        # predictions, references = accelerator.gather((predictions, outputs[2]))
        predictions, references = accelerator.gather((predictions, batch['labels']))

        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        test_metric.add_batch(
            predictions=predictions,
            references=references,
        )
        all_predictions = all_predictions + [*predictions.cpu().numpy()] 
        all_references = all_references + [*references.cpu().numpy()]
        all_confidences = all_confidences + [*confidences.cpu().numpy()]

    test_eval_metric = test_metric.compute()
    logger.info(f"test results: {test_eval_metric}")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            if args.task_name is not None:
                json.dump({"eval_accuracy": eval_metric["accuracy"],
                        "eval_f1": eval_metric["f1"],
                        "test_accuracy": test_eval_metric["accuracy"],
                        "test_f1": test_eval_metric["f1"]}, 
                        f, indent=4)
            else:
                json.dump(
                    {
                        'eval': eval_metric,
                        'test': test_eval_metric
                    }
                        , f, indent=4)

if __name__ == "__main__":
    main()
