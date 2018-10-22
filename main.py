# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/9/4 下午2:20
#   南京大学软件学院 Nanjing University Software Institute
#
import preprocess
import model
import tensorflow as tf
import os
def train_ABS():
    epoche = 50
    source_name = 'analyse_result.txt'
    checkpoint_dir = os.path.abspath('./checkpoint_ABS')

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    processor = preprocess.Preprocessor()
    m = model.Model()
    ops = m.train()
    meta = {
        'C':m.C,
        'V':m.V
    }
    data_g = processor.data_format_train(source_name,m.BATCH_SIZE,format_type='ABS',meta=meta)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        start_epoch = 0
        if checkpoint:
            saver.restore(sess,checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务'%checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        try:
            for i in range(epoche):
                try:
                    batch_count = 0
                    while True:
                        in_x,yc,y = next(data_g)
                        _,nll,gx = sess.run([ops['train_op'],ops['nll'],ops['gx']], feed_dict={ops['in_x']:in_x,ops['in_y']:y,ops['cont_y']:yc})
                        print('[INFO] Batch %d 训练结果：    nll-  %.6f  '%(batch_count,nll))
                        #print('[INFO] Batch %d'%batch_count)

                        batch_count+=1
                except StopIteration:
                    print("[INFO] Epoch %d 结束，现在开始保存模型..."%i)
                    saver.save(sess,os.path.join(checkpoint_dir, 'ABS_summary'), global_step=i)

        except KeyboardInterrupt:
            saver.save(sess, os.path.join(checkpoint_dir, 'ABS_summary'), global_step=i)
def valid_gen_ABS():
    processor = preprocess.Preprocessor()
    m = model.Model()
    ops = m.validation()
    checkpoint_dir = os.path.abspath('./checkpoint_ABS')
    source_name = 'analyse_result.txt'
    data_g = processor.data_format_eval(source_name,'ABS',{'V':m.V})

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 开始进行生成验证' % checkpoint)

            try:
                in_x, y = next(data_g)
                y_next= 2
                while y_next != 1:
                    res = [2]
                    pos = 0
                    y_c = preprocess.Preprocessor.context(res,pos,m.C)
                    nll, y_next = sess.run([ ops['nll'], ops['y_gen']],
                                          feed_dict={ops['in_x']: in_x, ops['in_y']: y, ops['cont_y']: y_c})
                    res.append(y_next)
            except StopIteration:
                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                saver.save(sess, os.path.join(checkpoint_dir, 'ABS_summary'), global_step=i)

def train_GEFG():
    epoch = 50
    source_name = 'analyse_result.txt'
    checkpoint_dir = os.path.abspath('./checkpoint_GEFG')

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    p = preprocess.Preprocessor()
    with tf.device('/cpu:0'):
        m = model.gated_evidence_fact_generation()
        ops = m.build_model('train')
    meta = {
        'MEL':m.MAX_EVID_LEN,
        'MEC':m.MAX_EVIDS,
        'MFL':m.MAX_FACT_LEN
    }
    data_gen = p.data_format_train(source_name,1,format_type="GEFG",meta=meta)
    t_op = m.train_op(ops['nll'])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        start_epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        try:
            for i in range(epoch):
                try:
                    batch_count = 0
                    while True:
                        evid_mat,evid_len,evid_count,fact_mat,fact_len = next(data_gen)
                        print(evid_len)
                        # oos,ss = sess.run([ops['os'],ops['ss']],feed_dict={ops['evid_mat']: evid_mat,
                        #                                  ops['evid_len']: evid_len,
                        #                                  ops['evid_count']: evid_count,
                        #                                  ops['fact_mat']: fact_mat,
                        #                                  ops['fact_len']: fact_len})
                        state_seq,output_seq,nll,_ = sess.run([ops['state_seq'],ops['output_seq'],ops['nll'],t_op],
                                              feed_dict={ops['evid_mat']: evid_mat,
                                                         ops['evid_len']: evid_len,
                                                         ops['evid_count']: evid_count,
                                                         ops['fact_mat']: fact_mat,
                                                         ops['fact_len']: fact_len}
                                              )

                        print('[INFO] Batch %d 训练结果：    nll-  %.6f  ' % (batch_count, nll))
                        # print('[INFO] Batch %d'%batch_count)

                        batch_count += 1
                except StopIteration:
                    print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                    saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)

        except KeyboardInterrupt:
            saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)

train_GEFG()