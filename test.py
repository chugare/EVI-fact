# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/8/17 下午2:08
#   南京大学软件学院 Nanjing University Software Institute
#

import tensorflow as tf
import numpy as np
import json
import re
import preprocess as PP
import model
np = PP.Preprocessor(False)
GEFG = model.gated_evidence_fact_generation()
dg = np.data_provider('train_data.json',{
        'NAME':'GEFG',
        'MEL':GEFG.MAX_EVID_LEN,
        'MEC':GEFG.MAX_EVIDS,
        'MFL':GEFG.MAX_FACT_LEN,
        'BATCH_SIZE':1
    })
c = 0
for i in dg:
    # print(c)
    # print(i[2])
    # print(i[4])
    # print('-------')
    c += 1



def t1():
    with tf.Session() as sess:
        seql = range(5)
        seqv = [i*5 for i in range(5)]

        evid_mat = tf.placeholder(dtype=tf.int32, shape=[4,5])
        evid_len = tf.placeholder(dtype=tf.int32, shape=[4])
        evid_count = tf.placeholder(dtype=tf.int32)
        i = tf.constant(0)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(200)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(200)
        print(fw_cell.state_size)

        embed_t = tf.get_variable(name='lookup',shape=[10000,2],initializer=tf.constant_initializer(2.0))
        evid_mat_embed = tf.nn.embedding_lookup(embed_t,evid_mat)

        state_ta = tf.TensorArray(dtype=tf.float32,size=evid_count)
        output_ta = tf.TensorArray(dtype=tf.float32,size=evid_count)

        mat_data = [range(5) for i in range(4)]
        length = range(1,5)
        i = tf.constant(0)
        tmp = tf.get_variable('k',shape=[5,2],dtype=tf.float32)
        def _encoder_evid(i,vec_mat,state_ta):
            vec = vec_mat[i][:evid_len[i]]
            # vec = tf.pad(vec,[[0,6-evid_len[i]],[0,0]])
            # vec = tf.reshape(vec,[6,2])
            state_ta = state_ta.write(i,vec)
            print('123')
            tmpx = vec
            i = tf.add(i,1)
            return i,vec_mat,state_ta
        loop, _, state_ta = tf.while_loop(lambda i,vec_mat,state_ta: i < evid_count, _encoder_evid,[i,evid_mat_embed,state_ta])

        init = tf.global_variables_initializer()
        sess.run(init)
        r1 = sess.run(evid_mat_embed,feed_dict={evid_mat:mat_data,evid_len:length,evid_count:4})
        print(r1)
        with tf.control_dependencies([loop]):
            ta_t = state_ta.stack()
        i,r = sess.run([loop,ta_t],feed_dict={evid_mat:mat_data,evid_len:length,evid_count:4})
        print(i)
        print(r)