#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
import tensorflow as tf
import math
import numpy as np
from ome import OME

class Model(object):
    def __init__(self, memory_size=512,memory_dim=100,shift_range=1,hidden_units=100, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.memory_size=memory_size
        self.memory_dim=memory_dim
        self.shift_range=shift_range
        self.hidden_units=hidden_units
        self.starting=tf.placeholder(tf.bool)

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.mask_type = tf.placeholder(dtype=tf.float32)
        self.mask_behavior_type = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.alias_type = tf.placeholder(dtype=tf.int32)
        self.alias_behavior_type = tf.placeholder(dtype=tf.int32)
        self.item = tf.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.item_type = tf.placeholder(dtype=tf.int32)
        self.item_behavior_type = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)
        self.tar_type = tf.placeholder(dtype=tf.int32)
        self.tar_behavior_type = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.get_variable('nasr_w1', [2*self.out_size, 2*self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))#初始化为均匀分布
        self.nasr_w2 = tf.get_variable('nasr_w2', [2*self.out_size, 2*self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w3 = tf.get_variable('nasr_w3', [2 * self.out_size, 2*self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_behavior = tf.get_variable('nasr_behavior', [2 * self.out_size, 2 * self.out_size], dtype=tf.float32,
                                             initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, 2*self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [2*self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.u = tf.get_variable('u', [self.out_size, 2 * self.out_size], dtype=tf.float32,
                                 initializer=tf.zeros_initializer())

        self.nasr_w4 = tf.get_variable('inner_encoder', [2*self.out_size, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w5 = tf.get_variable('outer_encoder', [2*self.out_size, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w6 = tf.get_variable('state_encoder', [2*self.out_size, 1], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))

    def forward(self, re_embedding,re_embedding_type, train=True):#re_embedding=（100，7，100）
        rm = tf.reduce_sum(self.mask, 1)#self.mask中1的个数（即序列中的商品数） rm=[10,4,7,……]
        rm_type = tf.reduce_sum(self.mask_type, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm)-1], axis=1))#得到最后一个的item编号（因为从0开始，所以减一）
        last_id_type = tf.gather_nd(self.alias_type, tf.stack([tf.range(self.batch_size), tf.to_int32(rm_type) - 1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))#得到批次中每个序列中的最后一个商品的表征
        last_h_type = tf.gather_nd(re_embedding_type, tf.stack([tf.range(self.batch_size), last_id_type], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],#批次中每个序列的表征（100，16，100），16为训练集中序列的最大长度
                         axis=0)                                                           #batch_size*T*d
        seq_h_type = tf.stack([tf.nn.embedding_lookup(re_embedding_type[i], self.alias_type[i]) for i in range(self.batch_size)],
                         # 批次中每个序列的表征（100，16，100），16为训练集中序列的最大长度
                         axis=0)
        fin_state_behavior_type = tf.nn.embedding_lookup(self.embedding_behavior_type,
                                                         self.item_behavior_type)  # fin_state_behavior_type=(100,5,100),self.item_behavior_type=(100,5,100)
        re_embedding_behavior_type = tf.reshape(fin_state_behavior_type, [self.batch_size, -1, self.out_size])
        seq_h_behavior_type = tf.stack(
            [tf.nn.embedding_lookup(re_embedding_behavior_type[i], self.alias_behavior_type[i]) for i in
             range(self.batch_size)], axis=0)
        #将物品表征与行为类型表征进行拼接，然后使用RNN建模，得到用户的一般偏好u
        seq_behavior_hh = tf.concat(
            [tf.reshape(seq_h, [-1, self.out_size]), tf.reshape(seq_h_behavior_type, [-1, self.out_size])],
            1)  # 作为RNN的输入(4700,200)
        cell = tf.nn.rnn_cell.BasicRNNCell(2*self.hidden_size)
        state_output_behavior_type, seq_behavior = tf.nn.dynamic_rnn(cell, tf.reshape(seq_behavior_hh,[self.batch_size, -1,2*self.out_size]),
                                                                     initial_state=tf.zeros([self.batch_size, 2*self.out_size]))
        u = seq_behavior
        last_h=tf.concat([last_h,last_h_type],1)#将item与对应的类型拼接
        last = tf.matmul(last_h, self.nasr_w1)#w1*vn
        seq_hh=tf.concat([tf.reshape(seq_h, [-1, self.out_size]),tf.reshape(seq_h_type, [-1, self.out_size])],1)#拼接类型
        seq = tf.matmul(seq_hh, self.nasr_w2)#w2*vi（？，200）
        last = tf.reshape(last, [self.batch_size, 1, -1])#（100,1,200）
        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, 2*self.out_size]) + self.nasr_b)#（100，？，200）sigmoid(w1*vn+w2*vi+c)
        coef = tf.matmul(tf.reshape(m, [-1, 2*self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])#注意力权重
        b = self.embedding[1:]#商品表征列表
        b_type_embedding=tf.nn.embedding_lookup(self.embedding_type, self.c_sort_value)
        b_type_embedding = tf.reshape(b_type_embedding, [-1, self.out_size])
        b_type=b_type_embedding[1:]
        #b_type=self.embedding_type[1:]
        b_embedding = tf.concat([b, b_type], 1)
        seq_htype = tf.concat([seq_h, seq_h_type], 2)  #
        re_seq_htype = tf.matmul(tf.reshape(seq_htype, [-1, 2 * self.out_size]), self.nasr_w3)#（？，200）
        if not self.nonhybrid:#如果使用混合表征
            sg=tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * tf.reshape(re_seq_htype,[self.batch_size,-1,2*self.out_size]), 1)#全局表征（100，200）（100,200）
            sl=tf.reshape(last, [-1, 2*self.out_size])#局部表征（100,200）
            self.sg=sg
            self.sl=sl


            #添加OME模块
            # 初始化ntm_cell，用于读写memory
            self.ome_cell = OME(mem_size=(self.memory_size, self.memory_dim), shift_range=self.shift_range,
                                hidden_units=self.hidden_units)
            # 创建用于存放读写memory的state的placeholder

            self.memory_network_reads, self.memory_new_state = self.ome_cell(self.state, sg,     #atttention_proj,
                                                                             self.starting)  # 返回值为OME模块的输出和新的记忆矩阵。（括号里参数为mi和局部表征）
            memory_new_state=self.memory_new_state
            # 将局部表征、全局表征和OME模块的输出进行标准化（经过处理的数据符合标准正态分布）
            att_mean, att_var = tf.nn.moments(self.sg, axes=[1])  # 计算局部表征的均值和方差
            self.sg = (self.sg - tf.expand_dims(att_mean,1)) / tf.expand_dims(tf.sqrt(att_var + 1e-10), 1)  # 归一化？
            glo_mean, glo_var = tf.nn.moments(self.sl, axes=[1])
            self.sl = (self.sl - tf.expand_dims(glo_mean,1)) / tf.expand_dims(tf.sqrt(glo_var + 1e-10), 1)
            ntm_mean, ntm_var = tf.nn.moments(self.memory_network_reads, axes=[1])
            self.memory_network_reads = (self.memory_network_reads - tf.expand_dims(ntm_mean, 1)) / tf.expand_dims(
                tf.sqrt(ntm_var + 1e-10), 1)
            #使用门控机制权衡当前会话和邻居会话的重要性
            new_gate = tf.matmul(self.sg, self.nasr_w4) + \
                       tf.matmul(self.memory_network_reads, self.nasr_w5) + \
                       tf.matmul(self.sl, self.nasr_w6)#（100,1）
            new_gate = tf.nn.sigmoid(new_gate)  # ft 公式（13）
            #self.narm_representation = tf.concat((self.attentive_session_represention, self.global_session_representation), axis=1)  # 局部表征和全局表征进行拼接得到IME的输出
            #当前会话的表征
            ma = tf.concat([self.sg, self.sl], -1)  ## 局部表征和全局表征进行拼接得到IME的输出（100,400）
            self.memory_representation = tf.concat((self.memory_network_reads, self.memory_network_reads), axis=1)#（100,400）
            #会话的最终表示
            final_representation = new_gate * ma + (1 - new_gate) * self.memory_representation  # ct （100,400）
            #用户的偏好表示（会话的表示与用户一般偏好拼接）
            final_representation=tf.concat([final_representation,u],1)
            self.B = tf.get_variable('B', [6 * self.out_size, 2*self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(final_representation, self.B)
            logits = tf.matmul(y1, b_embedding, transpose_b=True)#计算推荐分数
        else:
            ma = tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * tf.reshape(re_seq_htype,[self.batch_size,-1,self.out_size]), 1)
            logits = tf.matmul(ma, b_embedding, transpose_b=True)
        loss = tf.reduce_mean(#交叉熵损失
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()#取出全局中所有的参数
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits,memory_new_state

    def run(self, fetches, tar,tar_type, tar_behavior_type,item,item_type, item_behavior_type,adj_in, adj_in_type, adj_in_behavior_type,adj_out, adj_out_type, adj_out_behavior_type,alias,alias_type,  alias_behavior_type,mask,mask_type,mask_behavior_type,state,starting=False):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.tar_type:tar_type,self.tar_behavior_type: tar_behavior_type,self.item: item, self.item_type:item_type,self.item_behavior_type: item_behavior_type,self.adj_in: adj_in,
                                                 self.adj_in_type: adj_in_type, self.adj_in_behavior_type: adj_in_behavior_type,self.adj_out: adj_out, self.adj_out_type: adj_out_type, self.adj_out_behavior_type: adj_out_behavior_type,self.alias: alias, self.alias_type:alias_type,self.alias_behavior_type: alias_behavior_type,self.mask: mask,self.mask_type:mask_type,self.mask_behavior_type: mask_behavior_type,self.state:state,self.starting:starting})


class GGNN(Model):
    def __init__(self,memory_size=512,memory_dim=100,shift_range=1,hidden_units=100,hidden_size=100, out_size=100, batch_size=300, n_node=None,n_node_type=None,n_node_behavior_type=None,
                 c_sort_value=[],lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN,self).__init__(memory_size,memory_dim,shift_range,hidden_units,hidden_size, out_size, batch_size, nonhybrid)#初始化
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))#初始化节点列表
        self.embedding_type = tf.get_variable(shape=[n_node_type, hidden_size], name='embedding_type', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))  # 初始化节点列表
        self.embedding_behavior_type = tf.get_variable(shape=[n_node_behavior_type, hidden_size], name='behavior_type',
                                                       dtype=tf.float32,
                                                       initializer=tf.random_uniform_initializer(-self.stdv,
                                                                                                 self.stdv))  # 初始化节点列表
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])#邻接矩阵（入边
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])#（出边）
        self.adj_in_type = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])  # 邻接矩阵（入边
        self.adj_out_type = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])  # （出边）
        self.adj_in_behavior_type = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])  # 邻接矩阵（入边
        self.adj_out_behavior_type = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])  # （出边）
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, 2*self.hidden_units])
        self.n_node = n_node
        self.n_node_type=n_node_type
        self.n_node_behavior_type = n_node_behavior_type
        self.c_sort_value=c_sort_value
        #self.c_sort_value=tf.placeholder(dtype=tf.int32)
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        # H=[W_in;W_out]
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_in_type = tf.get_variable('W_in_type', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_in_behavior_type = tf.get_variable('W_in_behavior_type', shape=[self.out_size, self.out_size],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in_type = tf.get_variable('b_in_type', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in_behavior_type = tf.get_variable('b_in_behavior_type', [self.out_size], dtype=tf.float32,
                                                  initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out_type = tf.get_variable('W_out_type', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out_behavior_type = tf.get_variable('W_out_behavior_type', [self.out_size, self.out_size],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out_type = tf.get_variable('b_out_type', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out_behavior_type = tf.get_variable('b_out_behavior_type', [self.out_size], dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('ggnn_model', reuse=None):
            a,b=self.ggnn()
            self.loss_train, _,self.memory_new_state= self.forward(a,b)
        with tf.variable_scope('ggnn_model', reuse=True):
            c,d=self.ggnn()
            self.loss_test, self.score_test, self.memory_new_state= self.forward(c,d, train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def ggnn(self):#利用门控图神经网络分别学习物品表征和类别表征
        fin_state = tf.nn.embedding_lookup(self.embedding,self.item)  # 查表得到一个序列中不重复节点的表征[v1,……,vn]，（假设一个批次的序列中，最大的不重复节点数为7）
        fin_state_type = tf.nn.embedding_lookup(self.embedding_type, self.item_type)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])                          # 节点的初始状态（100，7，100）
                fin_state_type = tf.reshape(fin_state_type, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),                  # （100，7，100）
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_in_type = tf.reshape(tf.matmul(tf.reshape(fin_state_type, [-1, self.out_size]),          # （100，7，100）
                                                         self.W_in_type) + self.b_in_type, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),                        # （100，7，100）
                                        self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                fin_state_out_type = tf.reshape(tf.matmul(tf.reshape(fin_state_type, [-1, self.out_size]),                                          # （100，7，100）
                              self.W_out_type) + self.b_out_type, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),                                                  # （100，7，200）
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                av_type = tf.concat([tf.matmul(self.adj_in_type, fin_state_in_type),                                   # （100，7，200）
                                     tf.matmul(self.adj_out_type, fin_state_out_type)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1,self.out_size]))                       # state_output=（100，7，100），fin_state=（100，100）
                state_output_type, fin_state_type = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av_type, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state_type, [-1, self.out_size]))

        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size]), tf.reshape(fin_state_type,
                                                                                       [self.batch_size, -1,
                                                                                        self.out_size])  # （100，7，100）


