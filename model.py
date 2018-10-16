# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/31 上午10:20
#   南京大学软件学院 Nanjing University Software Institute
#
import tensorflow as tf

class Model:

    def __init__(self,config = None):
        self.NUM_UNIT = 100
        self.H = 200    # 输入词向量维度大小
        self.V = 40000  # 词汇量
        self.D = 200    # 输出词向量维度大小
        self.C = 10     # 输出感受范围
        self.DC_LAYER = 2
        self.BATCH_SIZE = 10
        self.Q = 100
        self.LR = 0.001


    # 在ABS的论文之中，介绍了三种编码器，分别是词袋模型的编码器，CNN的编码器，以及Attention based的编码器
    #
    # 由于磁带模型的编码器没有任何的序列信息，所以改进的时候在AttentionBased的情况中添加了上下文的关系

    def embedding_x(self,x):
        F = tf.get_variable(name='embedding_F',shape=[self.V,self.H],initializer=tf.truncated_normal_initializer(stddev=0.2))
        Fx = tf.nn.embedding_lookup(F,x)
        return  Fx
    def embedding_y(self,y):
        G = tf.get_variable(name='embedding_G',shape = [self.V,self.D])
        Gy = tf.nn.embedding_lookup(G,y)
        return Gy
    def BOWEncoder(self,x,seq_length):
        Fx = self.embedding_x(x)
        xi = Fx/seq_length
        return  xi
    def ABSencoder(self,x,y):
        #
        Fx = self.embedding_x(x)
        Gy = self.embedding_y(y)
        P = tf.get_variable(name = 'P',shape=[self.H,self.D*self.C],initializer=tf.truncated_normal_initializer(stddev=0.2))
        Gy = tf.reshape(Gy,[self.BATCH_SIZE,self.D*self.C])
        p_r = tf.matmul(Fx,tf.reshape(tf.matmul(P,Gy,transpose_b=True),[self.BATCH_SIZE,self.H,1]))
        p = tf.nn.softmax(p_r,name='p')
        enc_abs = p*Fx
        return enc_abs
    def nnlm_build(self,x,yc):

        y = yc
        E = tf.get_variable(name = 'embedding_E',shape=[self.V,self.D])
        Ey = tf.nn.embedding_lookup(E,y)
        Ey = tf.reshape(Ey,[self.BATCH_SIZE,-1])
        U = tf.get_variable(name='U' , shape=[self.H,self.D*self.C],initializer=tf.truncated_normal_initializer(stddev = 0.2))
        W = tf.get_variable(name='W' , shape=[self.V,self.H],initializer=tf.truncated_normal_initializer(stddev = 0.2))
        V = tf.get_variable(name='V' , shape=[self.V,self.H],initializer=tf.truncated_normal_initializer(stddev = 0.2))
        b = tf.get_variable(name='h_b',shape=[self.V],initializer=tf.constant_initializer(value=0))
        b_enc = tf.get_variable(name='h_b-enc',shape=[self.V],initializer=tf.constant_initializer(value=0))
        enc = self.ABSencoder(x,y)
        h = tf.matmul(U,Ey,transpose_b=True)
        Vh = tf.add(tf.matmul(h,V,transpose_a=True,transpose_b=True),b)
        Wenc = enc*W

        Wenc = tf.add(tf.reduce_sum(Wenc,axis=2),b_enc)

        gx = Wenc+Vh
        return gx
    #   此函数返回的值在经过softmax之后被看作是每一个单词在当前上下文以及原文的条件下的概率
    def calc_nll(self,gx,y_t):
    #   使用的训练损失函数是NLL negative log-likehood，中心思想是每一次生成的概率分布中，正确单词的概率值，如果比较大那么说明该单词的概率比较小，和实际情况不相符
    #   这一步的输入gx是经过nnlm模型计算的概率分布，y_t则是正确的下一个值的字典中的序号
        y_ind = tf.range(0,self.BATCH_SIZE)
        y_ind = tf.reshape(y_ind,[self.BATCH_SIZE,1])
        y_t = tf.reshape(y_t,[self.BATCH_SIZE,-1])
        y_t = tf.concat([y_ind,y_t],1)
        nll = tf.gather_nd(gx,y_t)
        return  nll

    def train(self):
    #   不同的encoder由于输出向量的维度不同，W向量的大小也不同，基础的方式是使用abs的encoder
        input_x = tf.placeholder(
            dtype= tf.int32,shape=[self.BATCH_SIZE,self.V]
        )
        input_y = tf.placeholder(dtype= tf.int32,shape=[self.BATCH_SIZE])
        y_context = tf.placeholder(tf.int32,shape=[self.BATCH_SIZE,self.C])
        gx = self.nnlm_build(input_x,y_context)

        gx = tf.nn.softmax(gx)

        nll = self.calc_nll(gx,input_y)

        nll_v = -tf.reduce_mean(tf.log(nll))

        train_op = tf.train.AdamOptimizer(self.LR).minimize(nll_v)
        ops = {
            'in_x':input_x,
            'in_y':input_y,
            'cont_y':y_context,
            'train_op':train_op,
            'nll':nll_v,
            'gx':gx

        }
        return ops
    def validation(self):
        input_x = tf.placeholder(
            dtype=tf.int32, shape=[self.BATCH_SIZE, -1]
        )
        input_y = tf.placeholder(dtype=tf.int32, shape=[self.BATCH_SIZE, -1])
        y_context = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE, -1])
        gx = self.nnlm_build(input_x, y_context)
        y_gen = tf.argmax(gx,1)
        nll = self.calc_nll(gx, input_y)
        ops = {
            'in_x': input_x,
            'in_y': input_y,
            'cont_y': y_context,
            'y_gen':y_gen,
            'nll': nll
        }
        return ops


    def decoder(self,encoder_states,last_state,yi_1):

        ops = {}
        cells = []
        for i in range(self.DC_LAYER):
            dc_cell = tf.nn.rnn_cell.BasicLSTMCell
            cells.append(dc_cell)
        m_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        init_state = m_cell.zero_state(self.BATCH_SIZE)
        m_cell.call(inputs=input)
        ops['decoder_state'] = init_state
        ops['']
        return ops


class gated_evidence_fact_generation:
    #利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码

    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 2000
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 100

    def get_cells(self):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.NUM_UNIT)

        return [fw_cell,bw_cell]

    @staticmethod
    def BiLSTMencoder(cells,input,seqLength,state=None):
        fw_cell = cells[0]
        bw_cell = cells[1]

        __batch_size = 1
        if state is None :
            init_state_fw = fw_cell.zero_state(__batch_size,tf.float32)
            init_state_bw = bw_cell.zero_state(__batch_size,tf.float32)
        elif len(state) == 2:
            init_state_bw = state[1]
            init_state_fw = state[0]
        else:
            print('[INFO] Require state with size 2')
        output, en_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                            cell_bw=bw_cell,
                                                            input=input,
                                                            sequence_length=seqLength,
                                                            initial_state_fw=init_state_fw,
                                                            initial_state_bw=init_state_bw)

        bi_en_states = tf.concat(en_states, 2)
        output = tf.concat(output, 1)
        return bi_en_states, output
    def gated_choose(self,multi_states,evid_len,context_vec):
        #使用证据编码后的全部的隐层状态来计算gate值
        gate_value = tf.TensorArray(tf.float32)
        attention_var_s = tf.get_variable('Attention_w',dtype=tf.float32,shape=[self.NUM_UNIT])
        attention_var_c = tf.get_variable('Attention_c',dtype=tf.float32,shape=[self.DECODER_NUM_UNIT])
        i = tf.constant(0)
        def _gate_calc(state_seq,context_vec):
            gate_v = tf.maximum(tf.reduce_sum(state_seq*attention_var_s,1)) + tf.reduce_sum(attention_var_c*context_vec)
            return gate_v
        def _step(i,mul_states,context_vec,gate_value):
            state_vec = multi_states.read(i)
            gate_value = gate_value.write(_gate_calc(state_vec,context_vec))
            i = tf.add(i,1)
            return i,multi_states,context_vec,gate_value

        _,_,_,gate_value = tf.while_loop(lambda i,multi_states,context_vec,gate_value:i<evid_len[i],gate_choose,[i,multi_states,context_vec,gate_value])
        gate_value.stack()
        i = tf.argmax(gate_value)
        sen_vec = multi_states.read(i)
        return sen_vec,i
    def context_vec_gen(self,states):

    def build_model(self,ops):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式
        evid_mat = tf.placeholder(dtype=tf.int32,shape=[self.MAX_EVIDS,self.MAX_EVID_LEN])
        evid_len = tf.placeholder(dtype=tf.int32,shape=[self.MAX_EVIDS])
        evid_count = tf.placeholder(dtype=tf.int32)
        fact_mat = tf.placeholder(dtype=tf.int32,shape=[self.MAX_EVID_LEN])
        #可以设置直接从已有的词向量读入
        embedding_t = tf.get_variable('embedding_table',shape=[self.MAX_VOCA_SZIE,self.VEC_SIZE])
        evid_mat = tf.nn.embedding_lookup(embedding_t,evid_mat)
        i = tf.constant(0)

        cells = self.get_cells()

        def _encoder_evid(i,state_ta,output_ta):

            state,output = gated_evidence_fact_generation.BiLSTMencoder(cells,evid_mat[i],evid_len)
            state_ta = state_ta.write(state)
            output_ta = output_ta.write(output)
            i = tf.add(i,1)
            return i,state_ta,output_ta

        state_ta = tf.TensorArray(size=evid_count)
        output_ta = tf.TensorArray(dynamic_size=True)

        _,state_ta,output_ta = tf.while_loop(lambda i,state_ta,output_ta:i<evid_count,_encoder_evid,(i,state_ta,output_ta))

        context_vec = tf.constant(0,dtype=tf.float32,shape=[self.DECODER_NUM_UNIT])
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE,tf.float32)
        output_seq = tf.TensorArray(size=3000,dynamic_size=True)
        state_seq = tf.TensorArray(size=3000,dynamic_size=True)
        map_out_w = tf.get_variable('map_out',shape=[self.MAX_VOCA_SZIE,self.DECODER_NUM_UNIT],dtype=tf.float32,initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias',shape=[self.MAX_VOCA_SZIE],dtype=tf.float32,initializer=tf.constant_initializer(0))
        i = tf.constant(0)
        def _decoder_step(i,state_seq,generated_seq,run_state):
            choosed_state,index = self.gated_choose(state_ta,evid_len,context_vec)
            state,output = decoder_cell.call(output_ta[index],run_state)
            dis_v = map_out_w*output+map_out_b
            dis_v = tf.nn.softmax(dis_v)
            char_most_pro = tf.argmax(dis_v)
            state_seq = state_seq.write(i,state)
            generated_seq = generated_seq.write(i, char_most_pro)
            i = tf.add(i,1)
            return i,state_seq,generated_seq,state
        _,state_seq,output_seq,_ = tf.while_loop(lambda i,sq,oq,s:i)
