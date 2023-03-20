#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/17 5:40
# @Author : {ZM7}
# @File : main.py
# @Software: PyCharm

from __future__ import division
import numpy as np
from model import *
from utils import build_graph, Data, split_validation
import pickle
import argparse
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sampl/tianchi')
parser.add_argument('--method', type=str, default='ggnn', help='ggnn/gat/gcn')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')

parser.add_argument('--memory_size', type=int, default=128,help='.')#邻居的个数
parser.add_argument('--memory_dim', type=int, default=100, help='.')
parser.add_argument('--shift_range', type=int, default=1,help='.')
parser.add_argument('--hidden_units', type=int, default=100, help='Number of GRU hidden units. initial:100')
opt = parser.parse_args()
train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
# all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.dataset == 'diginetica':
    n_node = 43098
if opt.dataset == 'tianchi':#10万条（运行这个！
    n_node = 1812#2118#商品个数+1（0号，商品是从1开始编号的）
    n_node_type=426#457
    n_node_behavior_type = 5
if opt.dataset == '2019-Nov':
    n_node = 16175#商品个数+1（0号，商品是从1开始编号的）
    n_node_type = 98  # 457
    n_node_behavior_type = 5
if opt.dataset == '2019-Oct':
    n_node = 5319#商品个数+1（0号，商品是从1开始编号的）
    n_node_type = 63
    n_node_behavior_type = 5
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
elif opt.dataset == 'random_sample_diginetica':
    n_node = 1332
elif opt.dataset == 'sample':
    n_node = 310
"""
else:
    n_node = 310
"""
# g = build_graph(all_train_seq)
train_data = Data(train_data, sub_graph=True, method=opt.method, shuffle=True)
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)

train_data_seq=train_data.inputs
train_data_type=train_data.inputs_type
train_data_tar=train_data.targets
train_data_tar_type=train_data.targets_type
train_data_seq_1=[i for item in train_data_seq for i in item]
train_data_type_1=[i for item in train_data_type for i in item]
c1=dict(zip(train_data_seq_1,train_data_type_1))#训练集中商品对应的列表
c2=dict(zip(train_data_tar,train_data_tar_type))
c1.update(c2)
#c1_sort_value=list(dict(sorted(c1.items())).values())

test_data_seq=test_data.inputs
test_data_type=test_data.inputs_type
test_data_tar=test_data.targets
test_data_tar_type=test_data.targets_type
test_data_seq_1=[i for item in test_data_seq for i in item]
test_data_type_1=[i for item in test_data_type for i in item]
c11=dict(zip(test_data_seq_1,test_data_type_1))#训练集中商品对应的列表
c21=dict(zip(test_data_tar,test_data_tar_type))
c11.update(c21)
c1.update(c11)

#c2_sort_value=list(dict(sorted(c11.items())).values())
c_sort_value=list(dict(sorted(c1.items())).values())
model = GGNN(memory_size=opt.memory_size,memory_dim=opt.memory_dim,shift_range=opt.shift_range,hidden_units=opt.hidden_units,hidden_size=opt.hiddenSize, out_size=opt.hiddenSize, batch_size=opt.batchSize, n_node=n_node,n_node_type=n_node_type,n_node_behavior_type=n_node_behavior_type,
                 c_sort_value=c_sort_value,lr=opt.lr, l2=opt.l2,  step=opt.step, decay=opt.lr_dc_step * len(train_data.inputs) / opt.batchSize, lr_dc=opt.lr_dc,
                 nonhybrid=opt.nonhybrid)
print(opt)
best_result = [0, 0]
best_epoch = [0, 0]
for epoch in range(opt.epoch):
    print('epoch: ', epoch, '===========================================')
    slices = train_data.generate_batch(model.batch_size)#slices=[[0,……99],[100,……199],……[]]划分批次
    fetches = [model.opt, model.loss_train, model.global_step,model.memory_new_state]
    starting=True
    session_memory_state = np.random.normal(0, 0.05, size=[1, 2*opt.hidden_units])
    print('start training: ', datetime.datetime.now())
    loss_ = []
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets,adj_in_type, adj_out_type,alias_type,item_type,mask_type,targets_type,adj_in_behavior_type, adj_out_behavior_type,alias_behavior_type,item_behavior_type,mask_behavior_type,targets_behavior_type = train_data.get_slice(i)
        _, loss, _ ,session_memory_state= model.run(fetches, targets,targets_type,targets_behavior_type, item, item_type,item_behavior_type,adj_in, adj_in_type,adj_in_behavior_type, adj_out, adj_out_type, adj_out_behavior_type,alias,alias_type,alias_behavior_type,  mask,mask_type,mask_behavior_type,session_memory_state,starting)
        starting = False
        loss_.append(loss)
    loss = np.mean(loss_)
    slices = test_data.generate_batch(model.batch_size)
    print('start predicting: ', datetime.datetime.now())
    hit, mrr, test_loss_ = [], [],[]
    for i, j in zip(slices, np.arange(len(slices))):
        adj_in, adj_out, alias, item, mask, targets,adj_in_type, adj_out_type,alias_type,item_type,mask_type,targets_type,adj_in_behavior_type, adj_out_behavior_type,alias_behavior_type,item_behavior_type,mask_behavior_type,targets_behavior_type = test_data.get_slice(i)
        scores, test_loss,session_memory_state = model.run([model.score_test, model.loss_test,model.memory_new_state], targets,targets_type,targets_behavior_type, item, item_type,item_behavior_type,adj_in, adj_in_type, adj_in_behavior_type,adj_out, adj_out_type, adj_out_behavior_type,alias,alias_type, alias_behavior_type, mask,mask_type,mask_behavior_type,session_memory_state,starting)
        test_loss_.append(test_loss)#记录损失
        index = np.argsort(scores, 1)[:, -20:]
        for score, target in zip(index, targets):
            hit.append(np.isin(target - 1, score))#判断target - 1是否在score数组中
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (20-np.where(score == target - 1)[0][0]))
    hit = np.mean(hit)*100#命中率
    mrr = np.mean(mrr)*100
    test_loss = np.mean(test_loss_)
    if hit >= best_result[0]:
        best_result[0] = hit
        best_epoch[0] = epoch
    if mrr >= best_result[1]:
        best_result[1] = mrr
        best_epoch[1]=epoch
    print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'%
          (loss, test_loss, best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
