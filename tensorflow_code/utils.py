#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np
import tensorflow as tf


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois,all_usr_pois_type, all_usr_pois_behavior_type,item_tail):#all_usr_pois=[[1, 2], [1], [4], [6], [8, 9],……]
    us_lens = [len(upois) for upois in all_usr_pois]#训练集中每个序列的长度，us_lens=[2,1,1,1,2,……]
    len_max = max(us_lens)#获取训练集中最长的序列len_max=16
    us_lens_type = [len(upois) for upois in all_usr_pois_type]  # 训练集中每个序列的长度，us_lens=[2,1,1,1,2,……]
    len_max_type = max(us_lens_type)
    us_lens_behavior_type = [len(upois) for upois in all_usr_pois_behavior_type]  # 训练集中每个序列的长度，us_lens=[2,1,1,1,2,……]
    len_max_behavior_type = max(us_lens_behavior_type)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]#将每个序列长度都变为16（通过补0）：us_pois=[[1, 2，0,0,0,0,0,0,0,0,0,0,0,0,0,0],[],[],……]
    us_pois_type = [upois_type + item_tail * (len_max_type - le_type) for upois_type, le_type in zip(all_usr_pois_type, us_lens_type)]
    us_pois_behavior_type = [upois_behavior_type + item_tail * (len_max_behavior_type - le_behavior_type) for upois_behavior_type, le_behavior_type in zip(all_usr_pois_behavior_type, us_lens_behavior_type)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]#us_pois中有数字的都变为1，us_msks=[[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,……]，……]
    us_msks_type = [[1] * le_type + [0] * (len_max_type - le_type) for le_type in us_lens_type]
    us_msks_behavior_type = [[1] * le_behavior_type + [0] * (len_max_behavior_type - le_behavior_type) for le_behavior_type in us_lens_behavior_type]
    return us_pois, us_msks, us_pois_type,us_msks_type,us_pois_behavior_type,us_msks_behavior_type,len_max,len_max_type,len_max_behavior_type


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, sub_graph=False, method='ggnn', sparse=False, shuffle=False):
        inputs = data[0]#获取物品ID序列
        inputs_type=data[2]#获取物品类别序列
        inputs_behavior_type=data[4]#获取行为类型序列
        inputs, mask,inputs_type,mask_type, inputs_behavior_type,mask_behavior_type,len_max,len_max_type,len_max_behavior_type = data_masks(inputs,inputs_type, inputs_behavior_type,[0])#inputs=[[1, 2，0,0,0,0,0,0,0,0,0,0,0,0,0,0],[],[],……]
        self.inputs = np.asarray(inputs)#np.asarray（）将输入转为矩阵格式self.inputs:shape=(1205,16)
        self.inputs_type=np.asarray(inputs_type)
        self.inputs_behavior_type = np.asarray(inputs_behavior_type)
        self.mask = np.asarray(mask)#self.mask:shape=(1205,16)
        self.mask_type = np.asarray(mask_type)
        self.mask_behavior_type = np.asarray(mask_behavior_type)
        self.len_max = len_max#16
        self.len_max_type=len_max_type
        self.len_max_behavior_type = len_max_behavior_type
        self.targets = np.asarray(data[1])#标签shape=(1205,)
        self.targets_type=np.asarray(data[3])
        self.targets_behavior_type = np.asarray(data[6])
        self.length = len(inputs)#一共多少个序列
        self.length_type=len(inputs_type)
        self.length_behavior_type = len(inputs_behavior_type)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def generate_batch(self, batch_size):#每100个序列为1个批次
        if self.shuffle:#随机打乱
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.inputs_type=self.inputs_type[shuffled_arg]
            self.inputs_behavior_type = self.inputs_behavior_type[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.mask_type=self.mask_type[shuffled_arg]
            self.mask_behavior_type = self.mask_behavior_type[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.targets_type=self.targets_type[shuffled_arg]
            self.targets_behavior_type = self.targets_behavior_type[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1#批次数量
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices#slices=[[0,……99],[100,……199],……[]]

    def get_slice(self, index):#构建有向图（物品图和类别图），获得相应的邻接矩阵
        if 1:
            items, items_type,items_behavior_type,n_node,n_node_type, n_node_behavior_type,A_in, A_out, alias_inputs,A_in_type,A_out_type,alias_inputs_type,A_in_behavior_type,A_out_behavior_type,alias_inputs_behavior_type = [],[],[],[],[],[],[],[], [], [], [], [],[],[],[]
            for u_input in self.inputs[index]:
                n_node.append(len(np.unique(u_input)))#n_node=[7,6,3,……]存储一个批次中不重复节点数量
            for u_input_type in self.inputs_type[index]:
                n_node_type.append(len(np.unique(u_input_type)))
            for u_input_behavior_type in self.inputs_behavior_type[index]:
                n_node_behavior_type.append(len(np.unique(u_input_behavior_type)))
            max_n_node = np.max(n_node)#批次中最大的长度（不重复节点个数）若为14
            max_n_node_type = np.max(n_node_type)#4
            max_n_node_behavior_type = np.max(n_node_behavior_type)#5
            if self.method == 'ggnn':
                for u_input in self.inputs[index]:                                                  # self.inputs[0]=[199,199,0,0,0,0,0,0,0,0,0,0,0,0,0,0]：16维
                    node = np.unique(u_input)                                                        # node=[0,199]
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])                     # items=[0,199,0,0,0,0,0]
                    u_A = np.zeros((max_n_node, max_n_node))  # 初始化邻接矩阵max_n_node*max_n_node
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1  # 根据序列构建有向图
                    u_sum_in = np.sum(u_A, 0)  # 将u_A矩阵中所有行加到第一行
                    u_sum_in[np.where(u_sum_in == 0)] = 1  # 将u_sum_in中0值置为1，避免分母为0
                    u_A_in = np.divide(u_A, u_sum_in)  # 归一化后的入边矩阵
                    u_sum_out = np.sum(u_A, 1)
                    u_sum_out[np.where(u_sum_out == 0)] = 1
                    u_A_out = np.divide(u_A.transpose(), u_sum_out)  # 归一化后的出边矩阵

                    A_in.append(u_A_in)  # 一个批次中的入边矩阵
                    A_out.append(u_A_out)  # 一个批次中的出边矩阵
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])                           # 相当于给一个批次中每个节点重新编号了[[16维],……[]]长度为100
                for u_input_type in self.inputs_type[index]:                                           # self.inputs[0]=[199,199,0,0,0,0,0,0,0,0,0,0,0,0,0,0]：16维
                    node_type = np.unique(u_input_type)                                                # node=[0,199]
                    items_type.append(node_type.tolist() + (max_n_node_type - len(node_type)) * [0])  # items=[0,199,0,0,0,0,0]
                    u_A_type = np.zeros((max_n_node_type, max_n_node_type))                           # 初始化邻接矩阵max_n_node*max_n_node
                    for i in np.arange(len(u_input_type) - 1):
                        if u_input_type[i + 1] == 0:
                            break
                        u_type = np.where(node_type == u_input_type[i])[0][0]
                        v_type = np.where(node_type == u_input_type[i + 1])[0][0]
                        u_A_type[u_type][v_type] = 1  # 根据序列构建有向图
                    u_sum_in_type = np.sum(u_A_type, 0)  # 将u_A矩阵中所有行加到第一行
                    u_sum_in_type[np.where(u_sum_in_type == 0)] = 1  # 将u_sum_in中0值置为1，避免分母为0
                    u_A_in_type = np.divide(u_A_type, u_sum_in_type)  # 归一化后的入边矩阵
                    u_sum_out_type = np.sum(u_A_type, 1)
                    u_sum_out_type[np.where(u_sum_out_type == 0)] = 1
                    u_A_out_type = np.divide(u_A_type.transpose(), u_sum_out_type)  # 归一化后的出边矩阵

                    A_in_type.append(u_A_in_type)  # 一个批次中的入边矩阵
                    A_out_type.append(u_A_out_type)  # 一个批次中的出边矩阵
                    alias_inputs_type.append([np.where(node_type == i)[0][0] for i in u_input_type])  # 相当于给一个批次中每个节点重新编号了[[16维],……[]]长度为100
                for u_input_behavior_type in self.inputs_behavior_type[index]:  # self.inputs[0]=[199,199,0,0,0,0,0,0,0,0,0,0,0,0,0,0]：16维
                    node_behavior_type = np.unique(u_input_behavior_type)  # node=[0,199]
                    items_behavior_type.append(node_behavior_type.tolist() + (max_n_node_behavior_type - len(node_behavior_type)) * [0])  # items=[0,199,0,0,0,0,0]
                    u_A_behavior_type = np.zeros((max_n_node_behavior_type, max_n_node_behavior_type))  # 初始化邻接矩阵max_n_node*max_n_node
                    for i in np.arange(len(u_input_behavior_type) - 1):
                        if u_input_behavior_type[i + 1] == 0:
                            break
                        u_behavior_type = np.where(node_behavior_type == u_input_behavior_type[i])[0][0]
                        v_behavior_type = np.where(node_behavior_type == u_input_behavior_type[i + 1])[0][0]
                        u_A_behavior_type[u_behavior_type][v_behavior_type] = 1  # 根据序列构建有向图
                    u_sum_in_behavior_type = np.sum(u_A_behavior_type, 0)  # 将u_A矩阵中所有行加到第一行
                    u_sum_in_behavior_type[np.where(u_sum_in_behavior_type == 0)] = 1  # 将u_sum_in中0值置为1，避免分母为0
                    u_A_in_behavior_type = np.divide(u_A_behavior_type, u_sum_in_behavior_type)  # 归一化后的入边矩阵
                    u_sum_out_behavior_type = np.sum(u_A_behavior_type, 1)
                    u_sum_out_behavior_type[np.where(u_sum_out_behavior_type == 0)] = 1
                    u_A_out_behavior_type = np.divide(u_A_behavior_type.transpose(), u_sum_out_behavior_type)  # 归一化后的出边矩阵

                    A_in_behavior_type.append(u_A_in_behavior_type)  # 一个批次中的入边矩阵
                    A_out_behavior_type.append(u_A_out_behavior_type)  # 一个批次中的出边矩阵
                    alias_inputs_behavior_type.append([np.where(node_behavior_type == i)[0][0] for i in u_input_behavior_type])

                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index],A_in_type,A_out_type,alias_inputs_type,items_type,self.mask_type[index],self.targets_type[index],A_in_behavior_type,A_out_behavior_type,alias_inputs_behavior_type,items_behavior_type,self.mask_behavior_type[index],self.targets_behavior_type[index]
            elif self.method == 'gat':
                A_in = []
                A_out = []
                for u_input in self.inputs[index]:
                    node = np.unique(u_input)
                    items.append(node.tolist() + (max_n_node - len(node)) * [0])
                    u_A = np.eye(max_n_node)
                    for i in np.arange(len(u_input) - 1):
                        if u_input[i + 1] == 0:
                            break
                        u = np.where(node == u_input[i])[0][0]
                        v = np.where(node == u_input[i + 1])[0][0]
                        u_A[u][v] = 1
                    A_in.append(-1e9 * (1 - u_A))
                    A_out.append(-1e9 * (1 - u_A.transpose()))
                    alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
                return A_in, A_out, alias_inputs, items, self.mask[index], self.targets[index]

        else:
            return self.inputs[index], self.mask[index], self.targets[index]