# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/31 上午10:20
#   南京大学软件学院 Nanjing University Software Institute
#
import tensorflow as tf
import numpy as np
from Evaluate import ROUGE_eval
class Base_model:
    def set_meta(self,meta):
        for k in meta:
            self.__dict__[k] = meta[k]
    def build_model(self,mode):
    # 建立模型，返回所有的操作组成的字典
        pass
    def train_fun(self,sess,data_gen,ops,global_step):
    # 训练函数，用于训练框架中调用
        pass
    def inter_fun(self,sess,data_gen,ops):
        pass
class ABS_model(Base_model):
    def __init__(self, config=None):
        self.NUM_UNIT = 100
        self.H = 200  # 输入词向量维度大小
        self.V = 10000  # 词汇量
        self.D = 200  # 输出词向量维度大小
        self.C = 30  # 输出感受范围
        self.DC_LAYER = 2
        self.BATCH_SIZE = 50
        self.Q = 100
        self.LR = 0.001
        self.DECAY_STEP = 5

    # 在ABS的论文之中，介绍了三种编码器，分别是词袋模型的编码器，CNN的编码器，以及Attention based的编码器
    #
    # 由于磁带模型的编码器没有任何的序列信息，所以改进的时候在AttentionBased的情况中添加了上下文的关系

    def embedding_x(self, x):
        F = tf.get_variable(name='embedding_F', shape=[self.V, self.H],
                            initializer=tf.truncated_normal_initializer(stddev=0.2))
        Fx = tf.nn.embedding_lookup(F, x)
        return Fx

    def embedding_y(self, y):
        G = tf.get_variable(name='embedding_G', shape=[self.V, self.D])
        Gy = tf.nn.embedding_lookup(G, y)
        return Gy

    def BOWEncoder(self, x, seq_length):
        Fx = self.embedding_x(x)
        xi = Fx / seq_length
        return xi

    def ABSencoder(self, x, y):
        #
        Fx = self.embedding_x(x)
        Gy = self.embedding_y(y)
        P = tf.get_variable(name='P', shape=[self.H, self.D * self.C],
                            initializer=tf.truncated_normal_initializer(stddev=0.2))
        Gy = tf.reshape(Gy, [self.BATCH_SIZE, self.D * self.C])
        p_r = tf.matmul(Fx, tf.reshape(tf.matmul(P, Gy, transpose_b=True), [self.BATCH_SIZE, self.H, 1]))
        p = tf.nn.softmax(p_r, name='p')
        enc_abs = p * Fx
        return enc_abs

    def nnlm_build(self, x, yc):
        y = yc
        E = tf.get_variable(name='embedding_E', shape=[self.V, self.D])
        Ey = tf.nn.embedding_lookup(E, y)
        Ey = tf.reshape(Ey, [self.BATCH_SIZE, -1])
        U = tf.get_variable(name='U', shape=[self.H, self.D * self.C],
                            initializer=tf.truncated_normal_initializer(stddev=0.2))
        W = tf.get_variable(name='W', shape=[self.V, self.H], initializer=tf.truncated_normal_initializer(stddev=0.2))
        V = tf.get_variable(name='V', shape=[self.V, self.H], initializer=tf.truncated_normal_initializer(stddev=0.2))
        b = tf.get_variable(name='h_b', shape=[self.V], initializer=tf.constant_initializer(value=0))
        b_enc = tf.get_variable(name='h_b-enc', shape=[self.V], initializer=tf.constant_initializer(value=0))
        enc = self.ABSencoder(x, y)
        h = tf.matmul(U, Ey, transpose_b=True)
        Vh = tf.add(tf.matmul(h, V, transpose_a=True, transpose_b=True), b)
        Wenc = enc * W

        Wenc = tf.add(tf.reduce_sum(Wenc, axis=2), b_enc)

        gx = Wenc + Vh
        return gx

    #   此函数返回的值在经过softmax之后被看作是每一个单词在当前上下文以及原文的条件下的概率
    def calc_loss(self, gx, y_t):
        #   使用的训练损失函数是NLL negative log-likehood，中心思想是每一次生成的概率分布中，正确单词的概率值，如果比较大那么说明该单词的概率比较小，和实际情况不相符
        #   这一步的输入gx是经过nnlm模型计算的概率分布，y_t则是正确的下一个值的字典中的序号
        y_one_hot = tf.one_hot(y_t, self.V)



        y_ind = tf.range(0, self.BATCH_SIZE)
        y_ind = tf.reshape(y_ind, [self.BATCH_SIZE, 1])
        y_t = tf.reshape(y_t, [self.BATCH_SIZE, -1])
        y_t = tf.concat([y_ind, y_t], 1)
        nll = tf.gather_nd(gx, y_t)
        return nll

    def build_model(self,mode):
        #   不同的encoder由于输出向量的维度不同，W向量的大小也不同，基础的方式是使用abs的encoder
        if mode == 'valid':
            self.BATCH_SIZE = 1

        input_x = tf.placeholder(
            dtype=tf.int32, shape=[self.BATCH_SIZE, self.V]
        )
        input_y = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE])
        y_context = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, self.C])
        gx = self.nnlm_build(input_x, y_context)
        y_oh = tf.one_hot(input_y,self.V)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_oh,logits=gx)
        gx = tf.nn.softmax(gx)
        y_gen = tf.argmax(gx, 1)
        nll = self.calc_loss(gx, input_y)
        nll_v = -tf.reduce_mean(tf.log(nll))
        tf.summary.histogram("NLL",nll_v)
        train_op = tf.train.AdamOptimizer(self.LR).minimize(nll_v)
        merge = tf.summary.merge_all()
        ops = {
            'in_x': input_x,
            'in_y': input_y,
            'cont_y': y_context,
            'train_op': train_op,
            'nll': nll_v,
            'cross_entropy':cross_entropy,
            'gx': gx,
            'y_gen':y_gen,
            'merge':merge

        }
        return ops
    def train_fun(self,sess,data_gen,ops,global_step):
        in_x, yc, y = next(data_gen)
        _, nll, gx ,merge= sess.run([ops['train_op'], ops['nll'], ops['gx'],ops['merge']],
                              feed_dict={ops['in_x']: in_x, ops['in_y']: y, ops['cont_y']: yc})
        res = {
            'loss':nll,
            'merge':merge
        }
        return res
    def inter_fun(self,sess,data_gen,ops):
        in_x, fact_vec = next(data_gen)
        yc = [0 for _ in range(self.C)]
        next_word = 2
        yc.append(2)
        yc = yc[1:]
        generate_seq = [2]
        yca = np.array(yc)
        yca = np.reshape(yc,[1,yca.shape[0]])
        count = 0
        while not next_word == 1 and count <400:
            nw , gx= sess.run([ ops['y_gen'],ops['gx']],
                              feed_dict={ops['in_x']: [in_x], ops['cont_y']: yca})

            yc.append(nw[0])
            yc = yc[1:]
            yca = np.array(yc)
            yca = np.reshape(yc, [1, yca.shape[0]])
            generate_seq.append(nw[0])
            count +=1
        res = {
            'out_seq':generate_seq,
            'fact_seq':fact_vec

        }
        return res

class gated_evidence_fact_generation(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码

    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 400
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 5
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20
    def get_cells(self):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)

        return [fw_cell, bw_cell]

    @staticmethod
    def BiLSTMencoder(cells, input_vec, seqLength, state=None):
        fw_cell = cells[0]
        bw_cell = cells[1]

        __batch_size = 1
        seqLength = tf.reshape(seqLength, [1])
        if state is None:
            init_state_fw = fw_cell.zero_state(__batch_size, tf.float32)
            init_state_bw = bw_cell.zero_state(__batch_size, tf.float32)
        elif len(state) == 2:
            init_state_bw = state[1]
            init_state_fw = state[0]
        else:
            print('[INFO] Require state with size 2')

        input_vec = tf.reshape(input_vec, [1, input_vec.shape[0], input_vec.shape[1]])
        output, en_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                            cell_bw=bw_cell,
                                                            inputs=input_vec,
                                                            sequence_length=seqLength,
                                                            initial_state_fw=init_state_fw,
                                                            initial_state_bw=init_state_bw)

        output = tf.concat(output, 2)
        state = tf.concat(en_states, 2)[0]

        return output, state

    def gated_choose(self, multi_states, evid_len, evid_count, context_vec):
        # 使用证据编码后的全部的隐层状态来计算gate值
        gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)
        attention_var_gate = tf.get_variable('attention_w', dtype=tf.float32,
                                          shape=[self.NUM_UNIT * 2, self.DECODER_NUM_UNIT * 2],
                                          initializer=tf.truncated_normal_initializer())
        i = tf.constant(0)

        attention_var_gen = tf.get_variable('attention_g',dtype=tf.float32,
                                            shape=[self.DECODER_NUM_UNIT*2,self.NUM_UNIT*2])

        def _gate_calc(state_seq, context_vec):
            state_seq = tf.reshape(state_seq, shape=[state_seq.shape[1], state_seq.shape[2]])
            context_vec = tf.reshape(context_vec, [-1])

            gate_v = tf.reduce_mean((tf.matmul(state_seq, attention_var_gate) * context_vec))
            return gate_v

        def _step(i, mul_states,out_seqs, context_vec, gate_value):
            state_vec = multi_states.read(i)
            out_seq = out_seqs.read(i)

            gate_value = gate_value.write(i, _gate_calc(state_vec, context_vec))


            i = tf.add(i, 1)
            return i, multi_states,out_seqs, context_vec, gate_value

        _, _, _, gate_value = tf.while_loop(lambda i, *_: i < evid_len[i], _step,
                                            [i, multi_states,out_seq, context_vec, gate_value], name='get_gate_value_loop')
        gate_value = gate_value.stack()
        i = tf.argmax(gate_value)
        i = tf.cast(i, tf.int32)
        sen_vec = multi_states.read(i)
        return sen_vec, i

    def context_vec_gen(self, states):
        pass

    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN])
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS])
        evid_count = tf.placeholder(dtype=tf.int32)
        fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN])
        fact_len = tf.placeholder(dtype=tf.int32)
        global_step = tf.placeholder(dtype=tf.int32)
        # 可以设置直接从已有的词向量读入
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        fact_mat_emb = tf.nn.embedding_lookup(embedding_t,fact_mat)
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r,self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)



        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT, state_is_tuple=True)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE, tf.float32)
        output_seq = tf.TensorArray(dtype=tf.int32, size=fact_len, clear_after_read=False)
        state_seq = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False)
        map_out_w = tf.get_variable('map_out', shape=[self.MAX_VOCA_SZIE, self.DECODER_NUM_UNIT], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias', shape=[self.MAX_VOCA_SZIE], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))
        nll = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False)
        gate_value = tf.TensorArray(dtype=tf.int32, size=fact_len, clear_after_read=False)

        attention_var_gate = tf.get_variable('attention_w', dtype=tf.float32,
                                             shape=[self.VEC_SIZE, self.DECODER_NUM_UNIT * 2],
                                             initializer=tf.truncated_normal_initializer())

        attention_var_gen = tf.get_variable('attention_g', dtype=tf.float32,
                                            shape=[ self.VEC_SIZE,self.DECODER_NUM_UNIT * 2])

        def _decoder_step(i, _state_seq, generated_seq, run_state, _gate_value, nll):




            context_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.DECODER_NUM_UNIT * 2]),
                                  lambda: _state_seq.read(tf.subtract(i, 1)))
            # 计算上下文向量直接使用上一次decoder的输出状态，作为上下文向量，虽然不一定好用，可能使用类似于ABS的上下文计算方式会更好，可以多试验
            i = tf.constant(0)
            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)
            def _gate_calc(state_seq, context_vec):
                state_seq = tf.reshape(state_seq, shape=[state_seq.shape[1], state_seq.shape[2]])
                context_vec = tf.reshape(context_vec, [-1])
                gate_v = tf.reduce_mean((tf.matmul(state_seq, attention_var_gate) * context_vec))
                align = tf.matmul(state_seq, attention_var_gen) * context_vec
                align_m  = tf.nn.softmax(align)
                content_vec = tf.reduce_sum(align_m * state_seq,0)
                return gate_v,content_vec
            def _step(i, out_seqs, context_vec, gate_value,attention_vec_evid):

                word_vec = evid_mat[i]

                gate_v,content_vec =_gate_calc(word_vec, context_vec)
                gate_value = gate_value.write(i,gate_v)
                attention_vec_evid = attention_vec_evid.write(i,content_vec)
                i = tf.add(i, 1)
                return i, multi_states, out_seqs, context_vec, gate_value,attention_vec_evid

            _, _, _, gate_value,attention_vec_evid = tf.while_loop(lambda i, *_: i < evid_len[i], _step,
                                                [i, multi_states, out_seq, context_vec, gate_value,attention_vec_evid],
                                                name='get_gate_value_loop')
            gate_value = gate_value.stack()
            i = tf.argmax(gate_value)
            i = tf.cast(i, tf.int32)
            sen_vec = multi_states.read(i)

            output, state = decoder_cell.apply(state_ta.read(index), run_state)



            # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
            mat_mul = map_out_w * output
            _gate_value = _gate_value.write(i,index)
            dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)

            char_most_pro = tf.argmax(dis_v)
            char_most_pro = tf.cast(char_most_pro, tf.int32)
            if mode == 'train':
            # 11/06 更改损失函数变为交叉熵
                true_l = tf.one_hot(fact_mat[i],depth=self.MAX_VOCA_SZIE)
                loss = tf.nn.softmax_cross_entropy_with_logits(dis_v,true_l,name='Cross_entropy')
                nll.write(i,loss)
                # dis_v = tf.nn.softmax(dis_v)
                # nll = nll.write(i, -tf.log(dis_v[fact_mat[i]]))
            # 对每一个单词的分布取最大值
            _state_seq = _state_seq.write(i, state)
            generated_seq = generated_seq.write(i, char_most_pro)
            # 生成context向量
            i = tf.add(i, 1)
            return i, _state_seq, generated_seq, state, _gate_value, nll

        _, state_seq, output_seq, run_state, gate_value, nll = tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                 [i, state_seq, output_seq, run_state, gate_value, nll],
                                                                 name='generate_word_loop')
        nll = nll.stack()
        gate_value = gate_value.stack()
        nll = tf.reduce_mean(nll)
        tf.summary.histogram('NLL', nll)
        merge = tf.summary.merge_all()
        state_seq = state_seq.stack()
        output_seq = output_seq.stack()
        e_lr = tf.train.exponential_decay(self.LR,global_step=global_step,decay_steps=self.DECAY_STEP,decay_rate=self.DECAY_RATE,staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)
        t_op = adam.minimize(nll)

        op = {
            'evid_mat': evid_mat_r,
            'evid_len': evid_len,
            'evid_count': evid_count,
            'fact_mat': fact_mat,
            'fact_len': fact_len,
            'global_step':global_step,
            'state_seq': state_seq,
            'output_seq': output_seq,
            'nll': nll,
            'merge': merge,
            'gate_value':gate_value,
            'train_op':t_op

        }

        return op

    def train_fun(self,sess,data_gen,ops,global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        state_seq, output_seq, nll, merge, _ = sess.run(
            [ops['state_seq'], ops['output_seq'], ops['nll'], ops['merge'], ops['train_op']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']:global_step}
            )

        return {
            'loss':nll,
            'merge':merge
        }


class gated_evidence_fact_generation_0(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码

    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 400
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 5
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20

    def get_cells(self):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)

        return [fw_cell, bw_cell]

    @staticmethod
    def BiLSTMencoder(cells, input_vec, seqLength, state=None):
        fw_cell = cells[0]
        bw_cell = cells[1]

        __batch_size = 1
        seqLength = tf.reshape(seqLength, [1])
        if state is None:
            init_state_fw = fw_cell.zero_state(__batch_size, tf.float32)
            init_state_bw = bw_cell.zero_state(__batch_size, tf.float32)
        elif len(state) == 2:
            init_state_bw = state[1]
            init_state_fw = state[0]
        else:
            print('[INFO] Require state with size 2')

        input_vec = tf.reshape(input_vec, [1, input_vec.shape[0], input_vec.shape[1]])
        output, en_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                            cell_bw=bw_cell,
                                                            inputs=input_vec,
                                                            sequence_length=seqLength,
                                                            initial_state_fw=init_state_fw,
                                                            initial_state_bw=init_state_bw)

        output = tf.concat(output, 2)
        state = tf.concat(en_states, 2)[0]

        return output, state

    def gated_choose(self, multi_states, evid_len, evid_count, context_vec):
        # 使用证据编码后的全部的隐层状态来计算gate值
        gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)
        attention_var_gate = tf.get_variable('attention_w', dtype=tf.float32,
                                             shape=[self.NUM_UNIT * 2, self.DECODER_NUM_UNIT * 2],
                                             initializer=tf.truncated_normal_initializer())
        i = tf.constant(0)

        attention_var_gen = tf.get_variable('attention_g', dtype=tf.float32,
                                            shape=[self.DECODER_NUM_UNIT * 2, self.NUM_UNIT * 2])

        def _gate_calc(state_seq, context_vec):
            state_seq = tf.reshape(state_seq, shape=[state_seq.shape[1], state_seq.shape[2]])
            context_vec = tf.reshape(context_vec, [-1])

            gate_v = tf.reduce_mean((tf.matmul(state_seq, attention_var_gate) * context_vec))
            return gate_v

        def _step(i, mul_states,  context_vec, gate_value):
            state_vec = mul_states.read(i)

            gate_value = gate_value.write(i, _gate_calc(state_vec, context_vec))

            i = tf.add(i, 1)
            return i, mul_states, out_seqs, context_vec, gate_value

        _, _, _, gate_value = tf.while_loop(lambda i, *_: i < evid_len[i], _step,
                                            [i, multi_states, context_vec, gate_value],
                                            name='get_gate_value_loop')
        gate_value = gate_value.stack()
        i = tf.argmax(gate_value)
        i = tf.cast(i, tf.int32)
        sen_vec = multi_states.read(i)
        return sen_vec, i

    def context_vec_gen(self, states):
        pass

    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN])
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS])
        evid_count = tf.placeholder(dtype=tf.int32)
        fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN])
        fact_len = tf.placeholder(dtype=tf.int32)
        global_step = tf.placeholder(dtype=tf.int32)
        # 可以设置直接从已有的词向量读入
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r, self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)
        i = tf.constant(0)

        cells = self.get_cells()

        def _encoder_evid(i, _state_ta, _output_ta):
            output, state = gated_evidence_fact_generation.BiLSTMencoder(cells, evid_mat[i], evid_len[i])
            _state_ta = _state_ta.write(i, state)
            _output_ta = _output_ta.write(i, output)

            i = tf.add(i, 1)
            return i, _state_ta, _output_ta

        state_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)
        output_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False)

        _, state_ta, output_ta = tf.while_loop(lambda i, state_ta, output_ta: i < evid_count, _encoder_evid,
                                               (i, state_ta, output_ta), name='get_lstm_encoder')

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT, state_is_tuple=True)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE, tf.float32)
        output_seq = tf.TensorArray(dtype=tf.int32, size=fact_len, clear_after_read=False)
        state_seq = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False)
        map_out_w = tf.get_variable('map_out', shape=[self.MAX_VOCA_SZIE, self.DECODER_NUM_UNIT], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias', shape=[self.MAX_VOCA_SZIE], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))
        i = tf.constant(0)
        nll = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False)
        gate_value = tf.TensorArray(dtype=tf.int32, size=fact_len, clear_after_read=False)

        def nnlm_context_calc(i):
            context_matrix = tf.cond(tf.less(i, self.CONTEXT_LEN),
                                     lambda: tf.pad(fact_mat_emb[:i], [[0, 0], [tf.sub(self.CONTEXT_LEN, i), 0]]),
                                     lambda: fact_mat_emb[tf.sub(i, self.CONTEXT_LEN):i])

        def _decoder_step(i, _state_seq, generated_seq, run_state, _gate_value, nll):

            context_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.DECODER_NUM_UNIT * 2]),
                                  lambda: _state_seq.read(tf.subtract(i, 1)))
            # 计算上下文向量直接使用上一次decoder的输出状态，作为上下文向量，虽然不一定好用，可能使用类似于ABS的上下文计算方式会更好，可以多试验



            choosed_state, index = self.gated_choose(output_ta, evid_len, evid_count, context_vec)
            output, state = decoder_cell.apply(state_ta.read(index), run_state)

            # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
            mat_mul = map_out_w * output
            _gate_value = _gate_value.write(i, index)
            dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)

            char_most_pro = tf.argmax(dis_v)
            char_most_pro = tf.cast(char_most_pro, tf.int32)
            if mode == 'train':
                # 11/06 更改损失函数变为交叉熵
                true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)
                loss = tf.nn.softmax_cross_entropy_with_logits(dis_v, true_l, name='Cross_entropy')
                nll.write(i, loss)
                # dis_v = tf.nn.softmax(dis_v)
                # nll = nll.write(i, -tf.log(dis_v[fact_mat[i]]))
            # 对每一个单词的分布取最大值
            _state_seq = _state_seq.write(i, state)
            generated_seq = generated_seq.write(i, char_most_pro)
            # 生成context向量
            i = tf.add(i, 1)
            return i, _state_seq, generated_seq, state, _gate_value, nll

        _, state_seq, output_seq, run_state, gate_value, nll = tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                             (i, state_seq, output_seq, run_state,
                                                                              gate_value, nll),
                                                                             name='generate_word_loop')
        nll = nll.stack()
        gate_value = gate_value.stack()
        nll = tf.reduce_mean(nll)
        tf.summary.histogram('NLL', nll)
        merge = tf.summary.merge_all()
        state_seq = state_seq.stack()
        output_seq = output_seq.stack()
        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)
        t_op = adam.minimize(nll)

        op = {
            'evid_mat': evid_mat_r,
            'evid_len': evid_len,
            'evid_count': evid_count,
            'fact_mat': fact_mat,
            'fact_len': fact_len,
            'global_step': global_step,
            'state_seq': state_seq,
            'output_seq': output_seq,
            'nll': nll,
            'merge': merge,
            'gate_value': gate_value,
            'train_op': t_op

        }

        return op

    def train_fun(self, sess, data_gen, ops, global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        state_seq, output_seq, nll, merge, _ = sess.run(
            [ops['state_seq'], ops['output_seq'], ops['nll'], ops['merge'], ops['train_op']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']: global_step}
        )

        return {
            'loss': nll,
            'merge': merge
        }