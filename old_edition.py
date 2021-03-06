
import  tensorflow as tf


class gated_evidence_fact_generation_5(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码
    # 第三版，修改目标，
    # 改变生成器方式，从lstm的方式变为神经语言模型的方式
    # 改变attention方式，采用矩阵乘法再进行截断的方式
    # 使用上一步的正确输出和attention向量的拼接作为lstm的每一步输入
    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 200
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 1
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20
        self.HIDDEN_SIZE = 300

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
    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN],name='evid_mat_r')
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS],name='evid_len')
        evid_count = tf.placeholder(dtype=tf.int32,name='evid_len')
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        if mode == 'train':
            fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN],name='fact_mat')
            fact_len = tf.placeholder(dtype=tf.int32,name='fact_len')
            global_step = tf.placeholder(dtype=tf.int32)
        else:
            fact_mat = tf.constant(0)
            fact_len = tf.constant(self.MAX_FACT_LEN)
            global_step = tf.constant(0)

        fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)

 # 可以设置直接从已有的词向量读入
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r,self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)

        cells = self.get_cells()

        def _encoder_evid(i, _state_ta, _output_ta):
            output, state = gated_evidence_fact_generation.BiLSTMencoder(cells, evid_mat[i], evid_len[i])
            _state_ta = _state_ta.write(i, state)
            _output_ta = _output_ta.write(i, output)

            i = tf.add(i, 1)
            return i, _state_ta, _output_ta

        state_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False, name='encoder_state_SEQ',
                                  tensor_array_name='ENC_STATE_TA')
        output_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False, name='encoder_output_SEQ',
                                   tensor_array_name='ENC_OUT_TA')
        i = tf.constant(1)
        _, state_ta, output_ta = tf.while_loop(lambda i, state_ta, output_ta: i < evid_count, _encoder_evid,
                                               (i, state_ta, output_ta), name='get_lstm_encoder')

        output_seq = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='OUTPUT_SEQ',tensor_array_name='OUTPUT_SQ_TA')

        gate_fc_w = tf.get_variable('gate',shape=[self.VEC_SIZE*2],dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer())
        loss_array = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='LOSS_COLLECTION',tensor_array_name='LOSS_TA')
        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)

        gate_value = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')
        min_loss_index = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')
        #选择机制所使用的attention向量
        attention_var_gate = tf.get_variable('attention_sel', dtype=tf.float32,
                                             shape=[self.VEC_SIZE*2, self.HIDDEN_SIZE],
                                             initializer=tf.glorot_normal_initializer())
        #生成器所使用的attention向量
        attention_var_gen = tf.get_variable('attention_gen', dtype=tf.float32,
                                            shape=[ self.VEC_SIZE*2,self.HIDDEN_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        #用于神经语言模型的生成器的向量
        U_GEN = tf.get_variable('U_GEN', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE,self.CONTEXT_LEN*self.VEC_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        U_ATTEN = tf.get_variable('U_ATTEN', dtype=tf.float32,
                                            shape=[self.HIDDEN_SIZE,self.CONTEXT_LEN*self.VEC_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        #
        GEN_b = tf.get_variable('Ub', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE],
                                            initializer=tf.constant_initializer(0))

        #
        W_GEN = tf.get_variable('W_GEN', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE,self.VEC_SIZE*2],
                                            initializer=tf.glorot_normal_initializer())

        #
        # V = tf.get_variable('V', dtype=tf.float32,
        #                                     shape=[self.HIDDEN_SIZE,self.CONTEXT_LEN*self.VEC_SIZE],
        #                                     initializer=tf.glorot_normal_initializer())
        # tf.summary.histogram('FACT',fact_mat)
        def _decoder_step(i, generated_seq, _gate_value,_min_loss_index, nll):

            if mode == 'train':
                content_mat = tf.cond(tf.less(i, self.CONTEXT_LEN),
                                  lambda: tf.pad(tf.slice(fact_mat_emb, [0, 0], [i, self.VEC_SIZE]),
                                                 [[self.CONTEXT_LEN - i, 0], [0, 0]]),
                                  lambda: tf.slice(fact_mat_emb, [i - self.CONTEXT_LEN, 0],
                                                   [self.CONTEXT_LEN, self.VEC_SIZE]), name="get_context")
            else:
                genseq = tf.cond(tf.equal(i,0),
                                 lambda : tf.zeros([self.MAX_FACT_LEN],dtype=tf.int32),
                                 lambda : tf.reshape(generated_seq.gather(tf.range(0,i)),[-1])
                                 )
                last_word_emb = tf.nn.embedding_lookup(params=embedding_t,ids=genseq)
                content_mat = tf.cond(tf.less(i,self.CONTEXT_LEN),
                                      lambda : tf.pad(tf.slice(last_word_emb,[0,0],[i,self.VEC_SIZE]),[[self.CONTEXT_LEN-i,0],[0,0]]),
                                      lambda : tf.slice(last_word_emb,[i-self.CONTEXT_LEN,0],[self.CONTEXT_LEN,self.VEC_SIZE]),name="get_context")
            U_CONTEXT_GEN  =  tf.reshape(tf.matmul(U_GEN,tf.reshape(content_mat,[-1,1])),[-1],name='U_CONTEXT_GEN')
            CONTEXT_ATTEN  =  tf.reshape(tf.matmul(U_ATTEN,tf.reshape(content_mat,[-1,1])),[-1],name='CONTEXT_ATTEN')



            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                        name='GATE_V_LOOP',tensor_array_name='GV_LP_TA')
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                                name='ATTENTION_VEC_LOOP',tensor_array_name='AT_VEC_LP_TA')
            loss_res_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                         name='LOSS_LOOP', tensor_array_name='LOSS_LP_TA')
            char_most_pro_ta = tf.TensorArray(dtype=tf.int32, size=evid_count, clear_after_read=False,
                                              name='CHAR_PRO_LOOP', tensor_array_name='CHAR_PRO_LP_TA')
            # 在训练的时候，由于每一个证据的关联性都不确定，所以我想把从每一个证据生成的新数据内容都进行训练和梯度下降


            _step_input = {
                'gate_value':gate_value,
                'attention_vec_evid':attention_vec_evid,
                'loss_res_ta':loss_res_ta,
                'char_most_pro_ta':char_most_pro_ta

            }
            if mode == 'train':
                true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)

            # tf.matmul(evid_mat,attention_var_gate)
            def _step(j,_step_input):

                gate_value = _step_input['gate_value']
                attention_vec_evid = _step_input['attention_vec_evid']
                loss_res_ta = _step_input['loss_res_ta']
                char_most_pro_ta = _step_input['char_most_pro_ta']


                word_vec_seq = output_ta.read(j)[0][0:evid_len[j]]

                gate_v = tf.matmul(word_vec_seq, attention_var_gate) * CONTEXT_ATTEN
                gate_m = tf.reduce_sum(gate_v,1)
                gate_m = tf.nn.softmax(gate_m)

                gate_m = tf.reshape(gate_m,[1,-1])

                gate_v = tf.reshape(tf.matmul(gate_m, word_vec_seq),[-1])
                gate_v = tf.nn.l2_normalize(gate_v)
                # max_word = tf.argmax(gate_m)
                # gate_v = word_vec_seq[max_word]
                gate_v = tf.reduce_mean(gate_v*gate_fc_w)

                align = tf.matmul(word_vec_seq, attention_var_gen) * CONTEXT_ATTEN
                align = tf.reduce_sum(align, 1)
                align = tf.reshape(align, [1, -1])
                align_m = tf.nn.softmax(align)
                content_vec = tf.reshape(tf.matmul(align_m, word_vec_seq), [-1])
                content_vec = tf.nn.l2_normalize(content_vec)
                gate_value = gate_value.write(j,gate_v)
                attention_vec_evid = attention_vec_evid.write(j,content_vec)


                if mode == 'train':
                    content_vec = tf.reshape(content_vec,[-1,1])
                    W_ENC = tf.reshape(tf.matmul(W_GEN,content_vec),[-1])
                    print(W_ENC)
                    print(U_CONTEXT_GEN)
                    print(GEN_b)
                    dis_v = tf.reduce_sum([W_ENC,U_CONTEXT_GEN,GEN_b],axis=0, name='dis_add_1')
                    # dis_v = tf.add(dis_v,GEN_b,name='dis_add_2')
                    # dis_v = W_ENC
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_v,labels= true_l, name='Cross_entropy')
                    char_most_pro = tf.cast(tf.argmax(dis_v), tf.int32)
                    loss_res_ta = loss_res_ta.write(j,loss)
                    char_most_pro_ta = char_most_pro_ta.write(j,char_most_pro)

                _step_output = {
                    'gate_value': gate_value,
                    'attention_vec_evid': attention_vec_evid,
                    'loss_res_ta': loss_res_ta,
                    'char_most_pro_ta':char_most_pro_ta
                }
                j = tf.add(j, 1)
                return j,_step_output
            _, _step_output = tf.while_loop(lambda j, *_: j < evid_count, _step,
                                                [tf.constant(0),_step_input],
                                                name='get_gate_value_loop')

            gate_value = _step_output['gate_value']
            attention_vec_evid = _step_output['attention_vec_evid']
            loss_res_ta = _step_output['loss_res_ta']
            char_most_pro_ta = _step_output['char_most_pro_ta']

            if mode == 'train':
            # 11/06 更改损失函数变为交叉熵
                total_loss = loss_res_ta.stack()
                char_most_pro_t = char_most_pro_ta.stack()
                next_state_i = tf.cast(tf.argmin(total_loss), tf.int32)
                generated_seq = generated_seq.write(i,char_most_pro_t[next_state_i])
                tl_sf = tf.one_hot(next_state_i,evid_count)
                gate_value = gate_value.stack()
                gate_value = gate_value[:evid_count]
                _gate_value = _gate_value.write(i, tf.cast(tf.argmax(gate_value),tf.int32))
                _min_loss_index = _min_loss_index.write(i,next_state_i)
                loss_g = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tl_sf,logits=gate_value)


                total_loss = loss_res_ta.read(next_state_i)+loss_g
            #
                nll = nll.write(i, total_loss)
            else:
                # 对每一个单词的分布取最大值
                gate_value = gate_value.stack()
                gate_index = tf.argmax(gate_value)
                gate_index = tf.cast(gate_index, tf.int32)
                _gate_value = _gate_value.write(i, gate_index)
                # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
                content_vec = attention_vec_evid.read(gate_index)
                content_vec = tf.reshape(content_vec,[-1,1])
                W_ENC = tf.matmul(W_GEN, content_vec)
                W_ENC = tf.reshape(W_ENC,[-1])
                dis_v = W_ENC + U_CONTEXT_GEN + GEN_b
                char_most_pro = tf.argmax(dis_v)
                generated_seq = generated_seq.write(i, tf.cast(char_most_pro,tf.int32))
            # 生成context向量
            i = tf.add(i, 1)

            return i, generated_seq, _gate_value,_min_loss_index,nll

        _,output_seq, gate_value,min_loss_index, nll= tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                 [tf.constant(0), output_seq,  gate_value,min_loss_index,loss_array],
                                                                 name='generate_word_loop')

        gate_value = gate_value.stack()
        # print(gate_value)
        tf.summary.histogram(name='GATE',values=gate_value)
        min_loss_index = min_loss_index.stack()
        if mode=='train':
            output_seq = output_seq.stack()
            tf.summary.histogram('OUT_PUT',output_seq[:fact_len])
            tc = tf.equal(output_seq,fact_mat)[:fact_len]
            accuracy = tf.reduce_sum(tf.cast(tc,tf.float32))/tf.cast(fact_len,tf.float32)
            tf.summary.scalar('PRECISION',accuracy)
            nll = nll.stack()
            fl = tf.cast(fact_len,dtype=tf.float32)
            nll = tf.reduce_sum(nll)/fl
            tf.summary.scalar('NLL', nll)
            # dis_v = distribute_ta.stack()
            # tf.summary.histogram("DIS_V",dis_v[:fact_len])
            grads = adam.compute_gradients(nll)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            # 使用直方图记录梯度
            for i,(grad, var) in enumerate(grads):
                if grad is not None:
                #     grads[i] = (tf.clip_by_norm(grad,5),var)
                # tf.summary.histogram(var.name + '/gradient', grads[i])
                    tf.summary.histogram(var.name + '/gradient', grad)

            t_op = adam.apply_gradients(grads)
        else:
            output_seq = output_seq.stack()
            t_op = tf.no_op()
            accuracy = tf.no_op()
        merge = tf.summary.merge_all()
        op = {
            'evid_mat': evid_mat_r,
            'evid_len': evid_len,
            'evid_count': evid_count,
            'fact_mat': fact_mat,
            'fact_len': fact_len,
            'global_step':global_step,
            'output_seq': output_seq,
            'nll': nll,
            'accuracy':accuracy,
            'merge': merge,
            'gate_value':gate_value,
            'min_loss_index':min_loss_index,
            'train_op':t_op
        }

        return op

    def train_fun(self,sess,data_gen,ops,global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)

        output_seq, nll,acc,gate_value,ml_index,merge,_ = sess.run(
            [ops['output_seq'],
             ops['nll'],
             ops['accuracy'],
             ops['gate_value'],
             ops['min_loss_index'],
             ops['merge'],
             ops['train_op']],
            feed_dict={
                       ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']:global_step
            }
            )
        # for i in evid_sig:
        #     print(i)
        # for i in dis_v:
        #     max_i = np.argmax(i)
        #     max_v = i[max_i]
        #     print("%d %f"%(max_i,max_v))
        # g_acc = 0
        # for i in range(fact_len):
        #     if gate_value[i] == ml_index[i]:
        #         g_acc += 1
        # g_acc = float(g_acc)/fact_len
        # print('[INFO-ex] Accuracy of gate_value : %.2f'%g_acc )
        #
        # gate_value_report_write('Gate_report.txt',evid_mat,fact_mat,ml_index)
        return {
            'loss':nll,
            'acc':acc,
            'merge':merge
        }
    def inter_fun(self,sess,data_gen,ops):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, gate_value= sess.run(
            [ops['output_seq'], ops['gate_value']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count}
        )
        c = 0
        for i in output_seq:
            if i != 1:
                c+=1

        output_seq = output_seq[:c]
        print(gate_value)
        return {
            'out_seq':output_seq,
            'fact_seq':fact_mat,
        }

class gated_evidence_fact_generation_4(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码
    # 第三版，修改目标，
    # 改变生成器方式，从lstm的方式变为神经语言模型的方式
    # 改变attention方式，采用矩阵乘法再进行截断的方式
    # 使用上一步的正确输出和attention向量的拼接作为lstm的每一步输入
    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 200
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 1
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20
        self.HIDDEN_SIZE = 300


    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN],name='evid_mat_r')
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS],name='evid_len')
        evid_count = tf.placeholder(dtype=tf.int32,name='evid_len')
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        if mode == 'train':
            fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN],name='fact_mat')
            fact_len = tf.placeholder(dtype=tf.int32,name='fact_len')
            global_step = tf.placeholder(dtype=tf.int32)
        else:
            fact_mat = tf.constant(0)
            fact_len = tf.constant(self.MAX_FACT_LEN)
            global_step = tf.constant(0)

        fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)

        # 可以设置直接从已有的词向量读入
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r,self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)

        output_seq = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='OUTPUT_SEQ',tensor_array_name='OUTPUT_SQ_TA')

        gate_fc_w = tf.get_variable('gate',shape=[self.VEC_SIZE],dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer())
        loss_array = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='LOSS_COLLECTION',tensor_array_name='LOSS_TA')
        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)

        gate_value = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')
        min_loss_index = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')
        #选择机制所使用的attention向量
        attention_var_gate = tf.get_variable('attention_sel', dtype=tf.float32,
                                             shape=[self.VEC_SIZE, self.HIDDEN_SIZE],
                                             initializer=tf.glorot_normal_initializer())
        #生成器所使用的attention向量
        attention_var_gen = tf.get_variable('attention_gen', dtype=tf.float32,
                                            shape=[ self.VEC_SIZE,self.HIDDEN_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        #用于神经语言模型的生成器的向量
        U_GEN = tf.get_variable('U_GEN', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE,self.CONTEXT_LEN*self.VEC_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        U_ATTEN = tf.get_variable('U_ATTEN', dtype=tf.float32,
                                            shape=[self.HIDDEN_SIZE,self.CONTEXT_LEN*self.VEC_SIZE],
                                            initializer=tf.glorot_normal_initializer())
        #
        GEN_b = tf.get_variable('Ub', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE],
                                            initializer=tf.constant_initializer(0))

        #
        W_GEN = tf.get_variable('W_GEN', dtype=tf.float32,
                                            shape=[self.MAX_VOCA_SZIE,self.VEC_SIZE],
                                            initializer=tf.glorot_normal_initializer())

        #
        # V = tf.get_variable('V', dtype=tf.float32,
        #                                     shape=[self.HIDDEN_SIZE,self.CONTEXT_LEN*self.VEC_SIZE],
        #                                     initializer=tf.glorot_normal_initializer())
        # tf.summary.histogram('FACT',fact_mat)
        def _decoder_step(i, generated_seq, _gate_value,_min_loss_index, nll):


            if mode == 'train':
                content_mat = tf.cond(tf.less(i,self.CONTEXT_LEN),
                                  lambda : tf.pad(tf.slice(fact_mat_emb,[0,0],[i,self.VEC_SIZE]),[[self.CONTEXT_LEN-i,0],[0,0]]),
                                  lambda : tf.slice(fact_mat_emb,[i-self.CONTEXT_LEN,0],[self.CONTEXT_LEN,self.VEC_SIZE]),name="get_context")
            else:

                genseq = tf.cond(tf.equal(i,0),
                                 lambda : tf.zeros([self.MAX_FACT_LEN],dtype=tf.int32),
                                 lambda : tf.reshape(generated_seq.gather(tf.range(0,i)),[-1])
                                 )
                fact_mat_emb = tf.nn.embedding_lookup(params=embedding_t,ids=genseq)
                content_mat = tf.cond(tf.less(i,self.CONTEXT_LEN),
                                      lambda : tf.pad(tf.slice(fact_mat_emb,[0,0],[i,self.VEC_SIZE]),[[self.CONTEXT_LEN-i,0],[0,0]]),
                                      lambda : tf.slice(fact_mat_emb,[i-self.CONTEXT_LEN,0],[self.CONTEXT_LEN,self.VEC_SIZE]),name="get_context")
            U_CONTEXT_GEN  =  tf.reshape(tf.matmul(U_GEN,tf.reshape(content_mat,[-1,1])),[-1],name='U_CONTEXT_GEN')
            CONTEXT_ATTEN  =  tf.reshape(tf.matmul(U_ATTEN,tf.reshape(content_mat,[-1,1])),[-1],name='CONTEXT_ATTEN')



            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                        name='GATE_V_LOOP',tensor_array_name='GV_LP_TA')
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                                name='ATTENTION_VEC_LOOP',tensor_array_name='AT_VEC_LP_TA')
            loss_res_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                         name='LOSS_LOOP', tensor_array_name='LOSS_LP_TA')
            char_most_pro_ta = tf.TensorArray(dtype=tf.int32, size=evid_count, clear_after_read=False,
                                              name='CHAR_PRO_LOOP', tensor_array_name='CHAR_PRO_LP_TA')
            # 在训练的时候，由于每一个证据的关联性都不确定，所以我想把从每一个证据生成的新数据内容都进行训练和梯度下降


            _step_input = {
                'gate_value':gate_value,
                'attention_vec_evid':attention_vec_evid,
                'loss_res_ta':loss_res_ta,
                'char_most_pro_ta':char_most_pro_ta

            }
            if mode == 'train':
                true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)

            # tf.matmul(evid_mat,attention_var_gate)
            def _step(j,_step_input):

                gate_value = _step_input['gate_value']
                attention_vec_evid = _step_input['attention_vec_evid']
                loss_res_ta = _step_input['loss_res_ta']
                char_most_pro_ta = _step_input['char_most_pro_ta']


                word_vec_seq = evid_mat[j][0:evid_len[j]]

                gate_v = tf.matmul(word_vec_seq, attention_var_gate) * CONTEXT_ATTEN
                gate_m = tf.reduce_sum(gate_v,1)
                gate_m = tf.nn.softmax(gate_m)
                max_word = tf.argmax(gate_m)
                # gate_m = tf.reshape(gate_m,[1,-1])
                #
                # gate_v = tf.reshape(tf.matmul(gate_m, word_vec_seq),[-1])
                # gate_v = tf.nn.l2_normalize(gate_v)
                gate_v = word_vec_seq[max_word]
                gate_v = tf.reduce_mean(gate_v*gate_fc_w)

                align = tf.matmul(word_vec_seq, attention_var_gen) * CONTEXT_ATTEN
                align = tf.reduce_sum(align, 1)
                align = tf.reshape(align, [1, -1])
                align_m = tf.nn.softmax(align)
                content_vec = tf.reshape(tf.matmul(align_m, word_vec_seq), [-1])
                content_vec = tf.nn.l2_normalize(content_vec)
                gate_value = gate_value.write(j,gate_v)
                attention_vec_evid = attention_vec_evid.write(j,content_vec)


                if mode == 'train':
                    content_vec = tf.reshape(content_vec,[-1,1])
                    W_ENC = tf.reshape(tf.matmul(W_GEN,content_vec),[-1])
                    print(W_ENC)
                    print(U_CONTEXT_GEN)
                    print(GEN_b)
                    dis_v = tf.reduce_sum([W_ENC,U_CONTEXT_GEN,GEN_b],axis=0, name='dis_add_1')
                    # dis_v = tf.add(dis_v,GEN_b,name='dis_add_2')
                    # dis_v = W_ENC
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_v,labels= true_l, name='Cross_entropy')
                    char_most_pro = tf.cast(tf.argmax(dis_v), tf.int32)
                    loss_res_ta = loss_res_ta.write(j,loss)
                    char_most_pro_ta = char_most_pro_ta.write(j,char_most_pro)

                _step_output = {
                    'gate_value': gate_value,
                    'attention_vec_evid': attention_vec_evid,
                    'loss_res_ta': loss_res_ta,
                    'char_most_pro_ta':char_most_pro_ta
                }
                j = tf.add(j, 1)
                return j,_step_output
            _, _step_output = tf.while_loop(lambda j, *_: j < evid_count, _step,
                                                [tf.constant(0),_step_input],
                                                name='get_gate_value_loop')

            gate_value = _step_output['gate_value']
            attention_vec_evid = _step_output['attention_vec_evid']
            loss_res_ta = _step_output['loss_res_ta']
            char_most_pro_ta = _step_output['char_most_pro_ta']

            if mode == 'train':
            # 11/06 更改损失函数变为交叉熵
                total_loss = loss_res_ta.stack()
                char_most_pro_t = char_most_pro_ta.stack()
                next_state_i = tf.cast(tf.argmin(total_loss), tf.int32)
                generated_seq = generated_seq.write(i,char_most_pro_t[next_state_i])
                tl_sf = tf.one_hot(next_state_i,evid_count)
                gate_value = gate_value.stack()
                gate_value = gate_value[:evid_count]
                _gate_value = _gate_value.write(i, tf.cast(tf.argmax(gate_value),tf.int32))
                _min_loss_index = _min_loss_index.write(i,next_state_i)
                loss_g = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tl_sf,logits=gate_value)


                total_loss = loss_res_ta.read(next_state_i)+loss_g
            #
                nll = nll.write(i, total_loss)
            else:
                # 对每一个单词的分布取最大值
                gate_value = gate_value.stack()
                gate_index = tf.argmax(gate_value)
                gate_index = tf.cast(gate_index, tf.int32)
                _gate_value = _gate_value.write(i, gate_index)
                # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
                content_vec = attention_vec_evid.read(gate_index)
                content_vec = tf.reshape(content_vec,[-1,1])
                W_ENC = tf.matmul(W_GEN, content_vec)
                W_ENC = tf.reshape(W_ENC,[-1])
                dis_v = W_ENC + U_CONTEXT_GEN + GEN_b
                char_most_pro = tf.argmax(dis_v)
                generated_seq = generated_seq.write(i, tf.cast(char_most_pro,tf.int32))
            # 生成context向量
            i = tf.add(i, 1)

            return i, generated_seq, _gate_value,_min_loss_index,nll

        _,output_seq, gate_value,min_loss_index, nll= tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                 [tf.constant(0), output_seq,  gate_value,min_loss_index,loss_array],
                                                                 name='generate_word_loop')

        gate_value = gate_value.stack()
        # print(gate_value)
        tf.summary.histogram(name='GATE',values=gate_value)
        min_loss_index = min_loss_index.stack()
        if mode=='train':
            output_seq = output_seq.stack()
            tf.summary.histogram('OUT_PUT',output_seq[:fact_len])
            tc = tf.equal(output_seq,fact_mat)[:fact_len]
            accuracy = tf.reduce_sum(tf.cast(tc,tf.float32))/tf.cast(fact_len,tf.float32)
            tf.summary.scalar('PRECISION',accuracy)
            nll = nll.stack()
            fl = tf.cast(fact_len,dtype=tf.float32)
            nll = tf.reduce_sum(nll)/fl
            tf.summary.scalar('NLL', nll)
            # dis_v = distribute_ta.stack()
            # tf.summary.histogram("DIS_V",dis_v[:fact_len])
            grads = adam.compute_gradients(nll)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            # 使用直方图记录梯度
            for i,(grad, var) in enumerate(grads):
                if grad is not None:
                #     grads[i] = (tf.clip_by_norm(grad,5),var)
                # tf.summary.histogram(var.name + '/gradient', grads[i])
                    tf.summary.histogram(var.name + '/gradient', grad)

            t_op = adam.apply_gradients(grads)
        else:
            output_seq = output_seq.stack()
            t_op = tf.no_op()
            accuracy = tf.no_op()
        merge = tf.summary.merge_all()
        op = {
            'evid_mat': evid_mat_r,
            'evid_len': evid_len,
            'evid_count': evid_count,
            'fact_mat': fact_mat,
            'fact_len': fact_len,
            'global_step':global_step,
            'output_seq': output_seq,
            'nll': nll,
            'accuracy':accuracy,
            'merge': merge,
            'gate_value':gate_value,
            'min_loss_index':min_loss_index,
            'train_op':t_op
        }

        return op

    def train_fun(self,sess,data_gen,ops,global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)

        output_seq, nll,acc,gate_value,ml_index,merge,_ = sess.run(
            [ops['output_seq'],
             ops['nll'],
             ops['accuracy'],
             ops['gate_value'],
             ops['min_loss_index'],
             ops['merge'],
             ops['train_op']],
            feed_dict={
                       ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']:global_step
            }
            )
        # for i in evid_sig:
        #     print(i)
        # for i in dis_v:
        #     max_i = np.argmax(i)
        #     max_v = i[max_i]
        #     print("%d %f"%(max_i,max_v))
        # g_acc = 0
        # for i in range(fact_len):
        #     if gate_value[i] == ml_index[i]:
        #         g_acc += 1
        # g_acc = float(g_acc)/fact_len
        # print('[INFO-ex] Accuracy of gate_value : %.2f'%g_acc )
        #
        # gate_value_report_write('Gate_report.txt',evid_mat,fact_mat,ml_index)
        return {
            'loss':nll,
            'acc':acc,
            'merge':merge
        }
    def inter_fun(self,sess,data_gen,ops):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, gate_value= sess.run(
            [ops['output_seq'], ops['gate_value']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count}
        )
        c = 0
        for i in output_seq:
            if i != 1:
                c+=1

        output_seq = output_seq[:c]
        print(gate_value)
        return {
            'out_seq':output_seq,
            'fact_seq':fact_mat,
        }

class gated_evidence_fact_generation_3(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码
    # 第三版，修改目标，
    # 改变生成器方式，从lstm的方式变为神经语言模型的方式
    # 改变attention方式，采用矩阵乘法再进行截断的方式
    # 使用上一步的正确输出和attention向量的拼接作为lstm的每一步输入
    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 200
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 1
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20



    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN],name='evid_mat_r')
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS],name='evid_len')
        evid_count = tf.placeholder(dtype=tf.int32,name='evid_len')
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        if mode == 'train':
            fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN],name='fact_mat')
            fact_len = tf.placeholder(dtype=tf.int32,name='fact_len')
            global_step = tf.placeholder(dtype=tf.int32)
        else:
            fact_mat = tf.constant(0)
            fact_len = tf.constant(self.MAX_FACT_LEN)
            global_step = tf.constant(0)

        fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)

        # 可以设置直接从已有的词向量读入
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r,self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)

        i =tf.constant(0)
        evid_sig = tf.TensorArray(dtype=tf.float32, size=self.MAX_EVIDS, clear_after_read=False,name='EVID_SIGINIFCANT',tensor_array_name='EVID_SIG')

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT, state_is_tuple=True)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE, tf.float32)
        state_seq = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='STATE_SEQ',tensor_array_name='STATE_SQ_TA')
        map_out_w = tf.get_variable('map_out', shape=[self.MAX_VOCA_SZIE, self.DECODER_NUM_UNIT], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias', shape=[self.MAX_VOCA_SZIE], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))


        gate_fc_w = tf.get_variable('gate',shape=[self.VEC_SIZE],dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer())
        loss_array = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='LOSS_COLLECTION',tensor_array_name='LOSS_TA')
        output_seq = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False, name='OUTPUT_SEQ',
                                    tensor_array_name='OUTPUT_SQ_TA')

        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)

        gate_value = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')
        min_loss_index = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')

        attention_var_gate = tf.get_variable('attention_sel', dtype=tf.float32,
                                             shape=[self.VEC_SIZE, self.DECODER_NUM_UNIT * 2],
                                             initializer=tf.glorot_normal_initializer())

        attention_var_gen = tf.get_variable('attention_gen', dtype=tf.float32,
                                            shape=[ self.VEC_SIZE,self.DECODER_NUM_UNIT * 2],
                                            initializer=tf.glorot_normal_initializer())

        tf.summary.histogram('FACT',fact_mat)
        # distribute_ta = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False,
        #                                 name='DIS_MAT',tensor_array_name='DIS_MAT_TA')
        def _decoder_step(i, _state_seq, generated_seq, run_state, _gate_value,_min_loss_index, nll):

            context_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.DECODER_NUM_UNIT * 2]),
                                  lambda: _state_seq.read(i-1))

            # 计算上下文向量直接使用上一次decoder的输出状态，作为上下文向量，虽然不一定好用，可能使用类似于ABS的上下文计算方式会更好，可以多试验

            # context_vec = tf.reshape(context_vec,[-1,1])
            # wc = tf.matmul(attention_var_gate,context_vec)
            # wc = tf.reshape(wc,[])
            # gate_value_after_attention = tf.tensordot(evid_mat,wc,2)
            # gate_value_after_attention = tf.reshape(gate_value_after_attention,[self.MAX_EVIDS,1,-1])
            #
            # gate_value_after_attention = tf.matmul(gate_value_after_attention,evid_mat)
            # gate_value_after_attention = tf.reshape(gate_value_after_attention,[self.MAX_EVIDS,-1])
            #
            # GV = tf.matmul(gate_value_after_attention,gate_fc_w)
            # GV = tf.reshape(GV,[-1])
            # GV = tf.softmax()
            #
            # wc_gen = tf.matmul(attention_var_gen,context_vec)
            # wc_gen = tf.reshape(wc_gen,[])
            # content_value_after_attention = tf.tensordot(evid_mat,wc_gen,2)
            # content_value_after_attention = tf.reshape(content_value_after_attention,[self.MAX_EVIDS,1,-1])
            #
            # content_value_after_attention = tf.matmul(content_value_after_attention,evid_mat)
            # content_value_after_attention = tf.reshape(content_value_after_attention,[self.MAX_EVIDS,-1])
            #
            # decoder_cell(content_value_after_attention,)
            #


            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                        name='GATE_V_LOOP',tensor_array_name='GV_LP_TA')
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                                name='ATTENTION_VEC_LOOP',tensor_array_name='AT_VEC_LP_TA')
            decoder_state_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                              name='DECODER_STATE_LOOP',tensor_array_name='DECODER_STATE_LP_TA')
            loss_res_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                         name='LOSS_LOOP', tensor_array_name='LOSS_LP_TA')
            char_most_pro_ta = tf.TensorArray(dtype=tf.int32, size=evid_count, clear_after_read=False,
                                              name='CHAR_PRO_LOOP', tensor_array_name='CHAR_PRO_LP_TA')
            # 在训练的时候，由于每一个证据的关联性都不确定，所以我想把从每一个证据生成的新数据内容都进行训练和梯度下降

            _step_input = {
                'decoder_state_ta':decoder_state_ta,
                'context_vec':context_vec,
                'gate_value':gate_value,
                'attention_vec_evid':attention_vec_evid,
                'loss_res_ta':loss_res_ta,
                'char_most_pro_ta':char_most_pro_ta
                # 'loss_index':loss_index,
                # 'total_loss_ta':total_loss_ta
            }



            def _gate_calc(word_vec_seq, context_vec):
                # word_vec_seq = tf.reshape(word_vec_seq, shape=[word_vec_seq.shape[1], word_vec_seq.shape[2]])

                context_vec = tf.reshape(context_vec, [-1])
                gate_v = tf.matmul(word_vec_seq, attention_var_gate) * context_vec

                gate_m = tf.reduce_sum(gate_v,1)
                gate_m = tf.nn.softmax(gate_m)
                gate_m = tf.reshape(gate_m,[1,-1])
                gate_v = tf.reshape(tf.matmul(gate_m, word_vec_seq),[-1])
                gate_v = tf.nn.l2_normalize(gate_v)
                gate_v = tf.reduce_mean(gate_v*gate_fc_w)

                align = tf.matmul(word_vec_seq, attention_var_gen) * context_vec
                align = tf.reduce_sum(align, 1)
                align = tf.reshape(align, [1, -1])
                align_m = tf.nn.softmax(align)
                content_vec = tf.reshape(tf.matmul(align_m, word_vec_seq), [-1])
                content_vec = tf.nn.l2_normalize(content_vec)
                return gate_v, content_vec

            def _step(j,_step_input):

                decoder_state_ta = _step_input['decoder_state_ta']
                context_vec = _step_input['context_vec']
                gate_value = _step_input['gate_value']
                attention_vec_evid = _step_input['attention_vec_evid']
                loss_res_ta = _step_input['loss_res_ta']
                char_most_pro_ta = _step_input['char_most_pro_ta']
                # loss_index = _step_input['loss_index']
                # total_loss_ta = _step_input['total_loss_ta']

                word_vec_seq = evid_mat[j][0:evid_len[j]]
                gate_v,content_vec =_gate_calc(word_vec_seq, context_vec)
                gate_value = gate_value.write(j,gate_v)
                attention_vec_evid = attention_vec_evid.write(j,content_vec)


                if mode == 'train':
                    last_word_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.VEC_SIZE]),
                                  lambda: fact_mat_emb[i-1])

                    content_vec = tf.concat(values=[content_vec, last_word_vec], axis=0)
                    content_vec = tf.reshape(content_vec, [1, -1])
                    decoder_output,decoder_state = decoder_cell(content_vec,run_state)
                    mat_mul = map_out_w * decoder_output
                    dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)


                    # with tf.device('/cpu'):
                    true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_v,labels= true_l, name='Cross_entropy')
                    char_most_pro = tf.cast(tf.argmax(dis_v), tf.int32)
                    # char_most_pro = tf.cast(fact_mat[i],tf.int32)
                    decoder_state_ta = decoder_state_ta.write(j,decoder_state)
                    loss_res_ta = loss_res_ta.write(j,loss)
                    char_most_pro_ta = char_most_pro_ta.write(j,char_most_pro)
                    # total_loss_ta = total_loss_ta.write(loss_index)

                _step_output = {
                    'decoder_state_ta': decoder_state_ta,
                    'context_vec': context_vec,
                    'gate_value': gate_value,
                    'attention_vec_evid': attention_vec_evid,
                    'loss_res_ta': loss_res_ta,
                    'char_most_pro_ta':char_most_pro_ta
                    # 'loss_index':loss_index,
                    # 'total_loss_ta':total_loss_ta
                }
                j = tf.add(j, 1)
                return j,_step_output
            _, _step_output = tf.while_loop(lambda j, *_: j < evid_count, _step,
                                                [tf.constant(0),_step_input],
                                                name='get_gate_value_loop')

            decoder_state_ta = _step_output['decoder_state_ta']
            context_vec = _step_output['context_vec']
            gate_value = _step_output['gate_value']
            attention_vec_evid = _step_output['attention_vec_evid']
            loss_res_ta = _step_output['loss_res_ta']
            char_most_pro_ta = _step_output['char_most_pro_ta']
            # loss_index = _step_output['loss_index']
            # total_loss_ta = _step_output['total_loss_ta']


            if mode == 'train':
            # 11/06 更改损失函数变为交叉熵
                total_loss = loss_res_ta.stack()
                char_most_pro_t = char_most_pro_ta.stack()
                next_state_i = tf.cast(tf.argmin(total_loss), tf.int32)
                generated_seq = generated_seq.write(i,char_most_pro_t[next_state_i])
                tl_sf = tf.one_hot(next_state_i,evid_count)

                gate_value = gate_value.stack()

                gate_value = gate_value[:evid_count]
                _gate_value = _gate_value.write(i, tf.cast(tf.argmax(gate_value),tf.int32))
                _min_loss_index = _min_loss_index.write(i,next_state_i)
                loss_g = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tl_sf,logits=gate_value)

                # content_vec = attention_vec_evid.read(next_state_i)
                # last_word_vec = tf.cond(tf.equal(i, 0),
                #                   lambda: tf.constant(0, dtype=tf.float32, shape=[self.VEC_SIZE]),
                #                   lambda: fact_mat_emb[i-1])

                run_state = decoder_state_ta.read(next_state_i)
                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0],run_state[1])
                # ec = tf.cast(evid_count,dtype=tf.float32)
                total_loss = loss_res_ta.read(next_state_i)+loss_g
                # 11/27 更改损失计算方式为从每一个证据生成进行计算
                # 12/12 更改损失计算方式为使用最低loss证据产生的loss计算
                # true_l = tf.one_hot(fact_mat[i],depth=self.MAX_VOCA_SZIE)
                # loss = tf.nn.softmax_cross_entropy_with_logits(dis_v,true_l,name='Cross_entropy')
                #
                nll = nll.write(i, total_loss)
                # dis_v = tf.nn.softmax(dis_v)

                # nll = nll.write(i, -tf.log(dis_v[fact_mat[i]]))
            else:
                # 对每一个单词的分布取最大值
                gate_value = gate_value.stack()
                gate_index = tf.argmax(gate_value)
                gate_index = tf.cast(gate_index, tf.int32)
                _gate_value = _gate_value.write(i, gate_index)
                # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
                content_vec = attention_vec_evid.read(gate_index)
                last_word_vec = tf.cond(tf.equal(i, 0),
                                        lambda: tf.constant(0, dtype=tf.float32, shape=[self.VEC_SIZE]),
                                        lambda: tf.nn.embedding_lookup(embedding_t,generated_seq.read(i-1)))

                content_vec = tf.concat(values=[content_vec, last_word_vec], axis=0)
                content_vec = tf.reshape(content_vec, [1, self.VEC_SIZE*2])

                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0], run_state[1])

                decoder_output, run_state = decoder_cell(content_vec, run_state)
                mat_mul = map_out_w * decoder_output
                dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)
                char_most_pro = tf.argmax(dis_v)
                generated_seq = generated_seq.write(i, tf.cast(char_most_pro,tf.int32))
            # 生成context向量
            _state_seq = _state_seq.write(i, run_state)
            i = tf.add(i, 1)

            return i, _state_seq, generated_seq, run_state, _gate_value,_min_loss_index,nll

        _, state_seq, output_seq, run_state, gate_value,min_loss_index, nll= tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                 [tf.constant(0), state_seq, output_seq, run_state, gate_value,min_loss_index,loss_array],
                                                                 name='generate_word_loop')

        gate_value = gate_value.stack()
        tf.summary.histogram(name='GATE',values=gate_value)
        min_loss_index = min_loss_index.stack()
        if mode=='train':
            output_seq = output_seq.stack()
            tf.summary.histogram('OUT_PUT',output_seq[:fact_len])
            tc = tf.equal(output_seq,fact_mat)[:fact_len]
            accuracy = tf.reduce_sum(tf.cast(tc,tf.float32))/tf.cast(fact_len,tf.float32)
            tf.summary.scalar('PRECISION',accuracy)
            nll = nll.stack()
            fl = tf.cast(fact_len,dtype=tf.float32)
            nll = tf.reduce_sum(nll)/fl
            tf.summary.scalar('NLL', nll)
            # dis_v = distribute_ta.stack()
            # tf.summary.histogram("DIS_V",dis_v[:fact_len])
            grads = adam.compute_gradients(nll)

            # for var in tf.trainable_variables():
            #     tf.summary.histogram(var.name, var)
            # # 使用直方图记录梯度
            # for i,(grad, var) in enumerate(grads):
            #     if grad is not None:
            #     #     grads[i] = (tf.clip_by_norm(grad,5),var)
            #     # tf.summary.histogram(var.name + '/gradient', grads[i])
            #         tf.summary.histogram(var.name + '/gradient', grad)

            t_op = adam.apply_gradients(grads)
        else:
            output_seq = output_seq.stack()
            t_op = tf.no_op()
            accuracy = tf.no_op()
        merge = tf.summary.merge_all()
        op = {
            'evid_mat': evid_mat_r,
            'evid_len': evid_len,
            'evid_count': evid_count,
            'fact_mat': fact_mat,
            'fact_len': fact_len,
            'global_step':global_step,
            'output_seq': output_seq,
            'nll': nll,
            'accuracy':accuracy,
            'merge': merge,
            'gate_value':gate_value,
            'min_loss_index':min_loss_index,
            'train_op':t_op
        }

        return op

    def train_fun(self,sess,data_gen,ops,global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, nll,acc,gate_value,ml_index,merge,evid_sig,_ = sess.run(
            [ops['output_seq'],
             ops['nll'],
             ops['accuracy'],
             ops['gate_value'],
             ops['min_loss_index'],
             ops['merge'],
             ops['train_op']],
            feed_dict={
                       ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']:global_step
            }
            )
        for i in evid_sig:
            print(i)
        # for i in dis_v:
        #     max_i = np.argmax(i)
        #     max_v = i[max_i]
        #     print("%d %f"%(max_i,max_v))
        # g_acc = 0
        # for i in range(fact_len):
        #     if gate_value[i] == ml_index[i]:
        #         g_acc += 1
        # g_acc = float(g_acc)/fact_len
        # print('[INFO-ex] Accuracy of gate_value : %.2f'%g_acc )
        #
        # gate_value_report_write('Gate_report.txt',evid_mat,fact_mat,ml_index)
        return {
            'loss':nll,
            'acc':acc,
            'merge':merge
        }
    def inter_fun(self,sess,data_gen,ops):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, gate_value= sess.run(
            [ops['output_seq'], ops['gate_value']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count}
        )
        c = 0
        for i in output_seq:
            if i != 1:
                c+=1

        output_seq = output_seq[:c]
        print(gate_value)
        return {
            'out_seq':output_seq,
            'fact_seq':fact_mat,
        }

class gated_evidence_fact_generation_2(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码
    # 第三版，修改目标，

    # 使用上一步的正确输出和attention向量的拼接作为lstm的每一步输入
    # 存在问题：引起了内存溢出
    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 100
        self.DECODER_NUM_UNIT = 200
        self.LR = 0.002
        self.OH_ENCODER = False
        self.DECAY_STEP = 5
        self.DECAY_RATE = 0.8
        self.CONTEXT_LEN = 20



    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN],name='evid_mat_r')
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS],name='evid_len')
        evid_count = tf.placeholder(dtype=tf.int32,name='evid_len')
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        if mode == 'train':
            fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN],name='fact_mat')
            fact_len = tf.placeholder(dtype=tf.int32,name='fact_len')
            global_step = tf.placeholder(dtype=tf.int32)
        else:
            fact_mat = tf.constant(0)
            fact_len = tf.constant(0)
            global_step = tf.constant(0)

        fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)

        # 可以设置直接从已有的词向量读入
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r,self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)



        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT, state_is_tuple=True)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE, tf.float32)
        output_seq = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='OUTPUT_SEQ',tensor_array_name='OUTPUT_SQ_TA')
        state_seq = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='STATE_SEQ',tensor_array_name='STATE_SQ_TA')
        map_out_w = tf.get_variable('map_out', shape=[self.MAX_VOCA_SZIE, self.DECODER_NUM_UNIT], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias', shape=[self.MAX_VOCA_SZIE], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))
        gate_fc_w = tf.get_variable('gate',shape=[self.VEC_SIZE],dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer())
        loss_array = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,name='LOSS_COLLECTION',tensor_array_name='LOSS_TA')
        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)

        gate_value = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False,name='GATE_VALUE',tensor_array_name='GV_TA')

        attention_var_gate = tf.get_variable('attention_sel', dtype=tf.float32,
                                             shape=[self.VEC_SIZE, self.DECODER_NUM_UNIT * 2],
                                             initializer=tf.glorot_normal_initializer())

        attention_var_gen = tf.get_variable('attention_gen', dtype=tf.float32,
                                            shape=[ self.VEC_SIZE,self.DECODER_NUM_UNIT * 2],
                                            initializer=tf.glorot_normal_initializer())

        tf.summary.histogram('FACT',fact_mat)
        distribute_ta = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False,
                                       name='DIS_MAT',tensor_array_name='DIS_MAT_TA')
        def _decoder_step(i, _state_seq, generated_seq, run_state, _gate_value, nll,dis_v_ta):

            context_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.DECODER_NUM_UNIT * 2]),
                                  lambda: _state_seq.read(i-1))

            # 计算上下文向量直接使用上一次decoder的输出状态，作为上下文向量，虽然不一定好用，可能使用类似于ABS的上下文计算方式会更好，可以多试验

            context_vec = tf.reshape(context_vec,[-1,1])
            wc = tf.matmul(attention_var_gate,context_vec)
            wc = tf.reshape(wc,[])
            gate_value_after_attention = tf.tensordot(evid_mat,wc,2)
            gate_value_after_attention = tf.reshape(gate_value_after_attention,[self.MAX_EVIDS,1,-1])
            gate_value_after_attention = tf.matmul(gate_value_after_attention,evid_mat)
            gate_value_after_attention = tf.reshape(gate_value_after_attention,[self.MAX_EVIDS,-1])




            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                        name='GATE_V_LOOP',tensor_array_name='GV_LP_TA')
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                                name='ATTENTION_VEC_LOOP',tensor_array_name='AT_VEC_LP_TA')
            decoder_state_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                              name='DECODER_STATE_LOOP',tensor_array_name='DECODER_STATE_LP_TA')
            loss_res_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                         name='LOSS_LOOP', tensor_array_name='LOSS_LP_TA')
            char_most_pro_ta = tf.TensorArray(dtype=tf.int32, size=evid_count, clear_after_read=False,
                                              name='CHAR_PRO_LOOP', tensor_array_name='CHAR_PRO_LP_TA')
            # 在训练的时候，由于每一个证据的关联性都不确定，所以我想把从每一个证据生成的新数据内容都进行训练和梯度下降

            _step_input = {
                'decoder_state_ta':decoder_state_ta,
                'context_vec':context_vec,
                'gate_value':gate_value,
                'attention_vec_evid':attention_vec_evid,
                'loss_res_ta':loss_res_ta,
                'char_most_pro_ta':char_most_pro_ta
                # 'loss_index':loss_index,
                # 'total_loss_ta':total_loss_ta
            }



            def _gate_calc(word_vec_seq, context_vec):
                # word_vec_seq = tf.reshape(word_vec_seq, shape=[word_vec_seq.shape[1], word_vec_seq.shape[2]])

                context_vec = tf.reshape(context_vec, [-1])
                gate_v = tf.matmul(word_vec_seq, attention_var_gate) * context_vec

                gate_m = tf.reduce_sum(gate_v,1)
                gate_m = tf.nn.softmax(gate_m)
                gate_m = tf.reshape(gate_m,[1,-1])
                gate_v = tf.reshape(tf.matmul(gate_m, word_vec_seq),[-1])
                gate_v = tf.nn.l2_normalize(gate_v)
                gate_v = tf.reduce_mean(gate_v*gate_fc_w)

                align = tf.matmul(word_vec_seq, attention_var_gen) * context_vec
                align = tf.reduce_sum(align, 1)
                align = tf.reshape(align, [1, -1])
                align_m = tf.nn.softmax(align)
                content_vec = tf.reshape(tf.matmul(align_m, word_vec_seq), [-1])
                content_vec = tf.nn.l2_normalize(content_vec)
                return gate_v, content_vec

            def _step(j,_step_input):

                decoder_state_ta = _step_input['decoder_state_ta']
                context_vec = _step_input['context_vec']
                gate_value = _step_input['gate_value']
                attention_vec_evid = _step_input['attention_vec_evid']
                loss_res_ta = _step_input['loss_res_ta']
                char_most_pro_ta = _step_input['char_most_pro_ta']
                # loss_index = _step_input['loss_index']
                # total_loss_ta = _step_input['total_loss_ta']

                word_vec_seq = evid_mat[j][0:evid_len[j]]
                gate_v,content_vec =_gate_calc(word_vec_seq, context_vec)
                gate_value = gate_value.write(j,gate_v)
                attention_vec_evid = attention_vec_evid.write(j,content_vec)


                if mode == 'train':
                    last_word_vec = tf.cond(tf.equal(i, 0),
                                            lambda: tf.constant(0, dtype=tf.float32, shape=[self.VEC_SIZE]),
                                            lambda: fact_mat_emb[i-1])

                    content_vec = tf.concat(values=[content_vec, last_word_vec], axis=0)
                    content_vec = tf.reshape(content_vec, [1, -1])
                    decoder_output,decoder_state = decoder_cell(content_vec,run_state)
                    mat_mul = map_out_w * decoder_output
                    print(mat_mul)
                    dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)


                    with tf.device('/cpu'):
                        true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_v,labels= true_l, name='Cross_entropy')
                    char_most_pro = tf.cast(tf.argmax(dis_v), tf.int32)
                    # char_most_pro = tf.cast(fact_mat[i],tf.int32)
                    decoder_state_ta = decoder_state_ta.write(j,decoder_state)
                    loss_res_ta = loss_res_ta.write(j,loss)
                    char_most_pro_ta = char_most_pro_ta.write(j,char_most_pro)
                    # total_loss_ta = total_loss_ta.write(loss_index)

                _step_output = {
                    'decoder_state_ta': decoder_state_ta,
                    'context_vec': context_vec,
                    'gate_value': gate_value,
                    'attention_vec_evid': attention_vec_evid,
                    'loss_res_ta': loss_res_ta,
                    'char_most_pro_ta':char_most_pro_ta
                    # 'loss_index':loss_index,
                    # 'total_loss_ta':total_loss_ta
                }
                j = tf.add(j, 1)
                return j,_step_output
            _, _step_output = tf.while_loop(lambda j, *_: j < evid_count, _step,
                                            [tf.constant(0),_step_input],
                                            name='get_gate_value_loop')

            decoder_state_ta = _step_output['decoder_state_ta']
            context_vec = _step_output['context_vec']
            gate_value = _step_output['gate_value']
            attention_vec_evid = _step_output['attention_vec_evid']
            loss_res_ta = _step_output['loss_res_ta']
            char_most_pro_ta = _step_output['char_most_pro_ta']
            # loss_index = _step_output['loss_index']
            # total_loss_ta = _step_output['total_loss_ta']


            if mode == 'train':
                # 11/06 更改损失函数变为交叉熵
                total_loss = loss_res_ta.stack()
                char_most_pro_t = char_most_pro_ta.stack()
                next_state_i = tf.cast(tf.argmin(total_loss), tf.int32)
                generated_seq = generated_seq.write(i,char_most_pro_t[next_state_i])
                tl_sf = tf.nn.softmax(tf.reciprocal(total_loss))
                gate_value = gate_value.stack()
                _gate_value = _gate_value.write(i, tf.cast(tf.argmax(gate_value),tf.int32))
                loss_g = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gate_value,labels=tl_sf)

                content_vec = attention_vec_evid.read(next_state_i)
                last_word_vec = tf.cond(tf.equal(i, 0),
                                        lambda: tf.constant(0, dtype=tf.float32, shape=[self.VEC_SIZE]),
                                        lambda: fact_mat_emb[i-1])

                content_vec = tf.concat(values = [content_vec,last_word_vec],axis=0)
                content_vec = tf.reshape(content_vec,[1,-1])
                decoder_output, run_state = decoder_cell(content_vec, run_state)
                dis_v = tf.add(tf.reduce_sum(map_out_w * decoder_output, 1), map_out_b)
                dis_v_ta = dis_v_ta.write(i,content_vec)
                # run_state = decoder_state_ta.read(next_state_i)
                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0],run_state[1])
                # ec = tf.cast(evid_count,dtype=tf.float32)
                total_loss = loss_res_ta.read(next_state_i)
                # 11/27 更改损失计算方式为从每一个证据生成进行计算
                # 12/12 更改损失计算方式为使用最低loss证据产生的loss计算
                # true_l = tf.one_hot(fact_mat[i],depth=self.MAX_VOCA_SZIE)
                # loss = tf.nn.softmax_cross_entropy_with_logits(dis_v,true_l,name='Cross_entropy')
                #
                nll = nll.write(i, total_loss)
                # dis_v = tf.nn.softmax(dis_v)

                # nll = nll.write(i, -tf.log(dis_v[fact_mat[i]]))
            else:
                # 对每一个单词的分布取最大值
                gate_value = gate_value.stack()
                gate_index = tf.argmax(gate_value)
                gate_index = tf.cast(gate_index, tf.int32)
                _gate_value = _gate_value.write(i, gate_index)


                # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
                content_vec = attention_vec_evid.read(gate_index)
                content_vec = tf.reshape(content_vec, [1, -1])
                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0], run_state[1])
                decoder_output, run_state = decoder_cell(content_vec, run_state)
                mat_mul = map_out_w * decoder_output
                dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)
                char_most_pro = tf.argmax(dis_v)
                generated_seq = generated_seq.write(i, char_most_pro)
            # 生成context向量
            _state_seq = _state_seq.write(i, run_state)
            i = tf.add(i, 1)

            return i, _state_seq, generated_seq, run_state, _gate_value,nll,dis_v_ta

        _, state_seq, output_seq, run_state, gate_value, nll,distribute_ta= tf.while_loop(lambda i, *_: i < fact_len, _decoder_step,
                                                                                          [tf.constant(0), state_seq, output_seq, run_state, gate_value,loss_array,distribute_ta],
                                                                                          name='generate_word_loop')

        gate_value = gate_value.stack()
        tf.summary.histogram(name='GATE',values=gate_value)
        state_seq = state_seq.stack()

        if mode=='train':
            output_seq = output_seq.stack()
            tf.summary.histogram('OUT_PUT',output_seq[:fact_len])
            tc = tf.equal(output_seq,fact_mat)[:fact_len]
            accuracy = tf.reduce_sum(tf.cast(tc,tf.float32))/tf.cast(fact_len,tf.float32)
            tf.summary.scalar('PRECISION',accuracy)
            nll = nll.stack()
            fl = tf.cast(fact_len,dtype=tf.float32)
            nll = tf.reduce_sum(nll)/fl
            tf.summary.scalar('NLL', nll)
            dis_v = distribute_ta.stack()
            tf.summary.histogram("DIS_V",dis_v[:fact_len])
            grads = adam.compute_gradients(nll)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            # 使用直方图记录梯度
            for i,(grad, var) in enumerate(grads):
                if grad is not None:
                    #     grads[i] = (tf.clip_by_norm(grad,5),var)
                    # tf.summary.histogram(var.name + '/gradient', grads[i])
                    tf.summary.histogram(var.name + '/gradient', grad)

            t_op = adam.apply_gradients(grads)
        else:
            output_seq = output_seq.stack()
            t_op = tf.no_op()
            accuracy = tf.no_op()
        merge = tf.summary.merge_all()
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
            'accuracy':accuracy,
            'merge': merge,
            'gate_value':gate_value,
            'train_op':t_op,
            "dis_v":dis_v
        }

        return op

    def train_fun(self,sess,data_gen,ops,global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        state_seq, output_seq, nll,acc, merge, _ ,dis_v= sess.run(
            [ops['state_seq'], ops['output_seq'], ops['nll'],ops['accuracy'], ops['merge'], ops['train_op'],ops["dis_v"]],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']:global_step}
        )
        # for i in dis_v:
        #     max_i = np.argmax(i)
        #     max_v = i[max_i]
        #     print("%d %f"%(max_i,max_v))
        return {
            'loss':nll,
            'acc':acc,
            'merge':merge
        }
    def inter_fun(self,sess,data_gen,ops):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, gate_value= sess.run(
            [ops['output_seq'], ops['gate_value']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count}
        )
        fact_vec = []
        for i in range(fact_len):
            fact_vec.append(fact_mat[i])


        return {
            'out_seq':output_seq,
            'fact_seq':fact_vec,
        }


class gated_evidence_fact_generation_1(Base_model):
    # 利用证据文本生成事实文本的模型，使用门控机制控制各个证据对于生成的作用，使用递归神经网络对每一个证据文本进行编码

    def __init__(self):
        self.NUM_UNIT = 100
        self.BATCH_SIZE = 1
        self.MAX_EVIDS = 50
        self.MAX_EVID_LEN = 800
        self.MAX_FACT_LEN = 600
        self.MAX_VOCA_SZIE = 10000
        self.VEC_SIZE = 200
        self.DECODER_NUM_UNIT = 200
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

    def build_model(self, mode):

        # encoder part
        # 将以tensor形式输入的原始数据转化成可变的TensorArray格式

        evid_mat_r = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS, self.MAX_EVID_LEN], name='evid_mat_r')
        evid_len = tf.placeholder(dtype=tf.int32, shape=[self.MAX_EVIDS], name='evid_len')
        evid_count = tf.placeholder(dtype=tf.int32, name='evid_len')
        embedding_t = tf.get_variable('embedding_table', shape=[self.MAX_VOCA_SZIE, self.VEC_SIZE],
                                      initializer=tf.truncated_normal_initializer())
        if mode == 'train':
            fact_mat = tf.placeholder(dtype=tf.int32, shape=[self.MAX_FACT_LEN], name='fact_mat')
            fact_len = tf.placeholder(dtype=tf.int32, name='fact_len')
            global_step = tf.placeholder(dtype=tf.int32)
        else:
            fact_mat = tf.constant(0)
            fact_len = tf.constant(0)
            global_step = tf.constant(0)
            # fact_mat_emb = tf.nn.embedding_lookup(embedding_t, fact_mat)

        # 可以设置直接从已有的词向量读入
        if self.OH_ENCODER:
            evid_mat = tf.one_hot(evid_mat_r, self.MAX_VOCA_SZIE)
        else:
            evid_mat = tf.nn.embedding_lookup(embedding_t, evid_mat_r)

        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.DECODER_NUM_UNIT, state_is_tuple=True)
        run_state = decoder_cell.zero_state(self.BATCH_SIZE, tf.float32)
        output_seq = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False, name='OUTPUT_SEQ',
                                    tensor_array_name='OUTPUT_SQ_TA')
        state_seq = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False, name='STATE_SEQ',
                                   tensor_array_name='STATE_SQ_TA')
        map_out_w = tf.get_variable('map_out', shape=[self.MAX_VOCA_SZIE, self.DECODER_NUM_UNIT], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
        map_out_b = tf.get_variable('map_bias', shape=[self.MAX_VOCA_SZIE], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))
        gate_fc_w = tf.get_variable('gate', shape=[self.VEC_SIZE], dtype=tf.float32,
                                    initializer=tf.glorot_normal_initializer())
        loss_array = tf.TensorArray(dtype=tf.float32, size=self.MAX_FACT_LEN, clear_after_read=False,
                                    name='LOSS_COLLECTION', tensor_array_name='LOSS_TA')
        e_lr = tf.train.exponential_decay(self.LR, global_step=global_step, decay_steps=self.DECAY_STEP,
                                          decay_rate=self.DECAY_RATE, staircase=False)
        adam = tf.train.AdamOptimizer(e_lr)

        gate_value = tf.TensorArray(dtype=tf.int32, size=self.MAX_FACT_LEN, clear_after_read=False, name='GATE_VALUE',
                                    tensor_array_name='GV_TA')

        attention_var_gate = tf.get_variable('attention_sel', dtype=tf.float32,
                                             shape=[self.VEC_SIZE, self.DECODER_NUM_UNIT * 2],
                                             initializer=tf.glorot_normal_initializer())

        attention_var_gen = tf.get_variable('attention_gen', dtype=tf.float32,
                                            shape=[self.VEC_SIZE, self.DECODER_NUM_UNIT * 2],
                                            initializer=tf.glorot_normal_initializer())

        tf.summary.histogram('FACT', fact_mat)
        distribute_ta = tf.TensorArray(dtype=tf.float32, size=fact_len, clear_after_read=False,
                                       name='DIS_MAT', tensor_array_name='DIS_MAT_TA')

        def _decoder_step(i, _state_seq, generated_seq, run_state, _gate_value, nll, dis_v_ta):

            context_vec = tf.cond(tf.equal(i, 0),
                                  lambda: tf.constant(0, dtype=tf.float32, shape=[self.DECODER_NUM_UNIT * 2]),
                                  lambda: _state_seq.read(i - 1))

            # 计算上下文向量直接使用上一次decoder的输出状态，作为上下文向量，虽然不一定好用，可能使用类似于ABS的上下文计算方式会更好，可以多试验

            gate_value = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                        name='GATE_V_LOOP', tensor_array_name='GV_LP_TA')
            attention_vec_evid = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                                name='ATTENTION_VEC_LOOP', tensor_array_name='AT_VEC_LP_TA')
            decoder_state_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                              name='DECODER_STATE_LOOP', tensor_array_name='DECODER_STATE_LP_TA')
            loss_res_ta = tf.TensorArray(dtype=tf.float32, size=evid_count, clear_after_read=False,
                                         name='LOSS_LOOP', tensor_array_name='LOSS_LP_TA')
            char_most_pro_ta = tf.TensorArray(dtype=tf.int32, size=evid_count, clear_after_read=False,
                                              name='CHAR_PRO_LOOP', tensor_array_name='CHAR_PRO_LP_TA')
            # 在训练的时候，由于每一个证据的关联性都不确定，所以我想把从每一个证据生成的新数据内容都进行训练和梯度下降

            _step_input = {
                'decoder_state_ta': decoder_state_ta,
                'context_vec': context_vec,
                'gate_value': gate_value,
                'attention_vec_evid': attention_vec_evid,
                'loss_res_ta': loss_res_ta,
                'char_most_pro_ta': char_most_pro_ta
                # 'loss_index':loss_index,
                # 'total_loss_ta':total_loss_ta
            }

            def _gate_calc(word_vec_seq, context_vec):
                # word_vec_seq = tf.reshape(word_vec_seq, shape=[word_vec_seq.shape[1], word_vec_seq.shape[2]])

                context_vec = tf.reshape(context_vec, [-1])
                gate_v = tf.matmul(word_vec_seq, attention_var_gate) * context_vec

                gate_m = tf.reduce_sum(gate_v, 1)
                gate_m = tf.nn.softmax(gate_m)
                gate_m = tf.reshape(gate_m, [1, -1])
                gate_v = tf.reshape(tf.matmul(gate_m, word_vec_seq), [-1])
                gate_v = tf.nn.l2_normalize(gate_v)
                gate_v = tf.reduce_mean(gate_v * gate_fc_w)

                align = tf.matmul(word_vec_seq, attention_var_gen) * context_vec
                align = tf.reduce_sum(align, 1)
                align = tf.reshape(align, [1, -1])
                align_m = tf.nn.softmax(align)
                content_vec = tf.reshape(tf.matmul(align_m, word_vec_seq), [-1])
                content_vec = tf.nn.l2_normalize(content_vec)
                return gate_v, content_vec

            def _step(j, _step_input):

                decoder_state_ta = _step_input['decoder_state_ta']
                context_vec = _step_input['context_vec']
                gate_value = _step_input['gate_value']
                attention_vec_evid = _step_input['attention_vec_evid']
                loss_res_ta = _step_input['loss_res_ta']
                char_most_pro_ta = _step_input['char_most_pro_ta']
                # loss_index = _step_input['loss_index']
                # total_loss_ta = _step_input['total_loss_ta']

                word_vec_seq = evid_mat[j][0:evid_len[j]]
                gate_v, content_vec = _gate_calc(word_vec_seq, context_vec)
                gate_value = gate_value.write(j, gate_v)
                attention_vec_evid = attention_vec_evid.write(j, content_vec)

                if mode == 'train':
                    content_vec = tf.reshape(content_vec, [1, -1])
                    decoder_output, decoder_state = decoder_cell(content_vec, run_state)
                    mat_mul = map_out_w * decoder_output
                    print(mat_mul)
                    dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)

                    with tf.device('/cpu'):
                        true_l = tf.one_hot(fact_mat[i], depth=self.MAX_VOCA_SZIE)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_v, labels=true_l, name='Cross_entropy')
                    char_most_pro = tf.cast(tf.argmax(dis_v), tf.int32)
                    # char_most_pro = tf.cast(fact_mat[i],tf.int32)
                    decoder_state_ta = decoder_state_ta.write(j, decoder_state)
                    loss_res_ta = loss_res_ta.write(j, loss)
                    char_most_pro_ta = char_most_pro_ta.write(j, char_most_pro)
                    # total_loss_ta = total_loss_ta.write(loss_index)

                _step_output = {
                    'decoder_state_ta': decoder_state_ta,
                    'context_vec': context_vec,
                    'gate_value': gate_value,
                    'attention_vec_evid': attention_vec_evid,
                    'loss_res_ta': loss_res_ta,
                    'char_most_pro_ta': char_most_pro_ta
                    # 'loss_index':loss_index,
                    # 'total_loss_ta':total_loss_ta
                }
                j = tf.add(j, 1)
                return j, _step_output

            _, _step_output = tf.while_loop(lambda j, *_: j < evid_count, _step,
                                            [tf.constant(0), _step_input],
                                            name='get_gate_value_loop')

            decoder_state_ta = _step_output['decoder_state_ta']
            context_vec = _step_output['context_vec']
            gate_value = _step_output['gate_value']
            attention_vec_evid = _step_output['attention_vec_evid']
            loss_res_ta = _step_output['loss_res_ta']
            char_most_pro_ta = _step_output['char_most_pro_ta']
            # loss_index = _step_output['loss_index']
            # total_loss_ta = _step_output['total_loss_ta']

            if mode == 'train':
                # 11/06 更改损失函数变为交叉熵
                total_loss = loss_res_ta.stack()
                char_most_pro_t = char_most_pro_ta.stack()
                next_state_i = tf.cast(tf.argmin(total_loss), tf.int32)
                generated_seq = generated_seq.write(i, char_most_pro_t[next_state_i])
                tl_sf = tf.nn.softmax(tf.reciprocal(total_loss))
                gate_value = gate_value.stack()
                _gate_value = _gate_value.write(i, tf.cast(tf.argmax(gate_value), tf.int32))
                loss_g = tf.nn.softmax_cross_entropy_with_logits_v2(logits=gate_value, labels=tl_sf)

                content_vec = attention_vec_evid.read(next_state_i)
                content_vec = tf.reshape(content_vec, [1, -1])
                decoder_output, run_state = decoder_cell(content_vec, run_state)
                dis_v = tf.add(tf.reduce_sum(map_out_w * decoder_output, 1), map_out_b)
                dis_v_ta = dis_v_ta.write(i, content_vec)
                # run_state = decoder_state_ta.read(next_state_i)
                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0], run_state[1])
                # ec = tf.cast(evid_count,dtype=tf.float32)
                total_loss = loss_res_ta.read(next_state_i)
                # 11/27 更改损失计算方式为从每一个证据生成进行计算
                # 12/12 更改损失计算方式为使用最低loss证据产生的loss计算
                # true_l = tf.one_hot(fact_mat[i],depth=self.MAX_VOCA_SZIE)
                # loss = tf.nn.softmax_cross_entropy_with_logits(dis_v,true_l,name='Cross_entropy')
                #
                nll = nll.write(i, total_loss)
                # dis_v = tf.nn.softmax(dis_v)

                # nll = nll.write(i, -tf.log(dis_v[fact_mat[i]]))
            else:
                # 对每一个单词的分布取最大值
                gate_value = gate_value.stack()
                gate_index = tf.argmax(gate_value)
                gate_index = tf.cast(gate_index, tf.int32)
                _gate_value = _gate_value.write(i, gate_index)

                # 生成的时候使用的是单层的lstm网络，每一个时间步生成一个向量，把这个向量放入全连接网络得到生成单词的分布
                content_vec = attention_vec_evid.read(gate_index)
                content_vec = tf.reshape(content_vec, [1, -1])
                run_state = tf.nn.rnn_cell.LSTMStateTuple(run_state[0], run_state[1])
                decoder_output, run_state = decoder_cell(content_vec, run_state)
                mat_mul = map_out_w * decoder_output
                dis_v = tf.add(tf.reduce_sum(mat_mul, 1), map_out_b)
                char_most_pro = tf.argmax(dis_v)
                generated_seq = generated_seq.write(i, char_most_pro)
            # 生成context向量
            _state_seq = _state_seq.write(i, run_state)
            i = tf.add(i, 1)

            return i, _state_seq, generated_seq, run_state, _gate_value, nll, dis_v_ta

        _, state_seq, output_seq, run_state, gate_value, nll, distribute_ta = tf.while_loop(lambda i, *_: i < fact_len,
                                                                                            _decoder_step,
                                                                                            [tf.constant(0), state_seq,
                                                                                             output_seq, run_state,
                                                                                             gate_value, loss_array,
                                                                                             distribute_ta],
                                                                                            name='generate_word_loop')

        gate_value = gate_value.stack()
        tf.summary.histogram(name='GATE', values=gate_value)
        state_seq = state_seq.stack()

        if mode == 'train':
            output_seq = output_seq.stack()
            tf.summary.histogram('OUT_PUT', output_seq[:fact_len])
            tc = tf.equal(output_seq, fact_mat)[:fact_len]
            accuracy = tf.reduce_sum(tf.cast(tc, tf.float32)) / tf.cast(fact_len, tf.float32)
            tf.summary.scalar('PRECISION', accuracy)
            nll = nll.stack()
            fl = tf.cast(fact_len, dtype=tf.float32)
            nll = tf.reduce_sum(nll) / fl
            tf.summary.scalar('NLL', nll)
            dis_v = distribute_ta.stack()
            tf.summary.histogram("DIS_V", dis_v[:fact_len])
            grads = adam.compute_gradients(nll)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            # 使用直方图记录梯度
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    #     grads[i] = (tf.clip_by_norm(grad,5),var)
                    # tf.summary.histogram(var.name + '/gradient', grads[i])
                    tf.summary.histogram(var.name + '/gradient', grad)

            t_op = adam.apply_gradients(grads)
        else:
            output_seq = output_seq.stack()
            t_op = tf.no_op()
            accuracy = tf.no_op()
        merge = tf.summary.merge_all()
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
            'accuracy': accuracy,
            'merge': merge,
            'gate_value': gate_value,
            'train_op': t_op,
            "dis_v": dis_v
        }

        return op

    def train_fun(self, sess, data_gen, ops, global_step):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        state_seq, output_seq, nll, acc, merge, _, dis_v = sess.run(
            [ops['state_seq'], ops['output_seq'], ops['nll'], ops['accuracy'], ops['merge'], ops['train_op'],
             ops["dis_v"]],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count,
                       ops['fact_mat']: fact_mat,
                       ops['fact_len']: fact_len,
                       ops['global_step']: global_step}
        )
        # for i in dis_v:
        #     max_i = np.argmax(i)
        #     max_v = i[max_i]
        #     print("%d %f"%(max_i,max_v))
        return {
            'loss': nll,
            'acc': acc,
            'merge': merge
        }

    def inter_fun(self, sess, data_gen, ops):
        evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
        output_seq, gate_value = sess.run(
            [ops['output_seq'], ops['gate_value']],
            feed_dict={ops['evid_mat']: evid_mat,
                       ops['evid_len']: evid_len,
                       ops['evid_count']: evid_count}
        )
        fact_vec = []
        for i in range(fact_len):
            fact_vec.append(fact_mat[i])

        return {
            'out_seq': output_seq,
            'fact_seq': fact_vec,
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

        def _step(i, mul_states, context_vec, gate_value):
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