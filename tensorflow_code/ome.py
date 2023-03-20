# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class OME():
    '''
    An OMECell that inherits from RNNCell. This inheritance was used to exploit
    the RNNCell's ability to be called by the dynamic_rnn() method, meaning
    that no custom code had to be implemented to perform dynamic unrollings of
    sequences with arbitrary lengths.
    '''
    def __init__(self, mem_size, shift_range=1, hidden_units=100):
        self.memory_size, self.memory_dim = mem_size
        self.shift_range = shift_range
        self.controller = tf.nn.rnn_cell.GRUCell(hidden_units)#
        self.hidden_units = hidden_units
        self._num_units = self.memory_dim*self.memory_size + 2*self.memory_size
        self.controller_hidden_layer_size = 100
        self.controller_layer_numbers = 0
        self.controller_output_size = self.memory_dim + 1 + 1 + (self.shift_range * 2 + 1) + 1 + self.memory_dim * 3 + 1 + 1 + (self.shift_range * 2 + 1) + 1

    @property
    def state_size(self):
        '''
        State includes the memory matrix, and address vectors for the read
        and write operations. These values influence the matrix and addresses at
        the next time step.
        '''
        # return self.memory_size * (self.memory_dim,) + (self.memory_size, self.memory_size)
        return (self.memory_size, self.memory_size)
    @property
    def output_size(self):
        '''
        Return only the size of the value that's read from the memory matrix.
        '''
        return self.memory_dim

    def __call__(self, memory_state, session_represention, starting, scope=None):
        def direct_assign():
            # 当每轮最开始时
            read_memory = session_represention#即全局表征（100,200）
            new_memory_state = session_represention#（100,200）
            return read_memory, new_memory_state

        def update_memory():
            # 求最近邻-余弦相似度
            cos_similarity = self.smooth_cosine_similarity(session_represention, memory_state)  # 计算相似度sim(clt,mi) [batch, n_session]
            neigh_sim, neigh_num = tf.nn.top_k(cos_similarity, k=self.memory_size)  # [batch_size, memory_size]#找出最相似的self.memory_size个邻居session的相似度值及最相似会话的索引索引
            session_neighborhood = tf.nn.embedding_lookup(memory_state,#查找出邻居session的表征
                                                          neigh_num)  # [batch_size, memory_size, memory_dim]
            neigh_sim = tf.expand_dims(tf.nn.softmax(neigh_sim), axis=2)#公式（10）计算邻居会话的权重
            read_memory = tf.reduce_sum(neigh_sim * session_neighborhood, axis=1)  # [batch_size, memory_dim]# 得到邻居会话的表征
            new_memory_state = tf.concat((memory_state, session_represention), axis=0)[-10000:]#更新记忆矩阵
            return read_memory, new_memory_state

        read_memory, new_memory_state = tf.cond(starting, direct_assign, update_memory)#如果starting为true，调用direct_assign，如果为false，调用update_memory

        return read_memory, new_memory_state

    def smooth_cosine_similarity(self, session_emb, sess_all_representations):
        """
        :param session_emb: a [batch_size*hidden_units] tensor
        :param sess_all_representations: a [n_session*hidden_units] tensor
        :return: a [batch_size*n_session] weighting vector
        """

        # Cosine Similarity
        sess_all_representations = tf.tile(tf.expand_dims(sess_all_representations, axis=0), multiples=[tf.shape(session_emb)[0], 1,1])#（100，？，200）
        session_emb = tf.expand_dims(session_emb, axis=2)#（100,200,1）
        inner_product = tf.matmul(sess_all_representations, session_emb)  # [batch_size,memory_size,1]  即相似度分子的计算点积（括号里参数为局部表征，记忆矩阵中的表征）
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(session_emb), axis=1, keepdims=True))#相似度公式中分母的计算
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(sess_all_representations), axis=2, keepdims=True))#相似度公式中分母的计算
        norm_product = M_norm * k_norm#相似度公式中分母的计算
        similarity = tf.squeeze(inner_product / (norm_product + 1e-8), axis=2)                                        #tf.squeeze默认删除所有为1的维度

        return similarity