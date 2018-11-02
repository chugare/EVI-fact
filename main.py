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
import time
import json



ERROR_LOG = open('error_log.txt','w',encoding='utf-8')


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
    # 设置训练配置内容
    epoch = 50
    source_name = 'train_data.json'
    checkpoint_dir = os.path.abspath('./checkpoint_GEFG')
    summary_dir = os.path.abspath('./summary/GEFG/train')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    p = preprocess.Preprocessor()


    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = model.gated_evidence_fact_generation()
    ops = m.build_model('train')

    # 配置数据生成器的元数据
    meta = {
        'MEL':m.MAX_EVID_LEN,
        'MEC':m.MAX_EVIDS,
        'MFL':m.MAX_FACT_LEN
    }
    t_op = m.train_op(ops['nll'])

    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        log_device_placement=True
    )
    with tf.Session(config=config) as sess:
        # 训练配置，包括参数初始化以及读取检查点
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess.graph.finalize()
        train_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        start_epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        start_time = time.time()

        # 开始训练
        global_step = 0
        for i in range(epoch):
            data_gen = p.data_format_train(source_name, 1, format_type="GEFG", meta=meta)
            try:
                batch_count = 0
                while True:
                    evid_mat,evid_len,evid_count,fact_mat,fact_len = next(data_gen)
                    print(fact_len)

                    last_time = time.time()
                    state_seq,output_seq,nll,merge,_ = sess.run([ops['state_seq'],ops['output_seq'],ops['nll'],ops['merge'],t_op],
                                          feed_dict={ops['evid_mat']: evid_mat,
                                                     ops['evid_len']: evid_len,
                                                     ops['evid_count']: evid_count,
                                                     ops['fact_mat']: fact_mat,
                                                     ops['fact_len']: fact_len},
                                          )

                    cur_time =time.time()
                    time_cost = cur_time-last_time
                    total_cost = cur_time-start_time

                    if global_step % 10 == 0:
                        train_writer.add_summary(merge,global_step/10)
                    print('[INFO] Batch %d 训练结果：NLL=%.6f  用时: %.2f 共计用时 %.2f' % (batch_count, nll,time_cost,total_cost))
                    # print('[INFO] Batch %d'%batch_count)
                    # matplotlib 实现可视化loss
                    batch_count += 1
                    global_step += 1
            except StopIteration:
                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)
            except Exception as e:
                print("[INFO] 因为程序错误停止训练，开始保存模型")
                saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)
            except KeyboardInterrupt:
                print("[INFO] 强行停止训练，开始保存模型")
                saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)

def valid_GEFG():

    source_name = 'test_data.json'
    checkpoint_dir = os.path.abspath('./checkpoint_GEFG')
    summary_dir = os.path.abspath('./summary/GEFG/test')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    p = preprocess.Preprocessor()

    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = model.gated_evidence_fact_generation()
    ops = m.build_model('test')

    # 配置数据生成器的元数据
    meta = {
        'MEL': m.MAX_EVID_LEN,
        'MEC': m.MAX_EVIDS,
        'MFL': m.MAX_FACT_LEN
    }


    checkpoint_dir = os.path.abspath('./checkpoint_GEFG')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 训练配置，包括参数初始化以及读取检查点
        # init = tf.global_variables_initializer()
        # sess.run(init)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess.graph.finalize()
        train_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        start_epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从训练的检查点 %s 载入模型进行验证' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        else:
            print('[ERROR] 不存在已经训练完成的模型，无法进行验证')
        start_time = time.time()
        global_step = 0

        report_data = []
        data_gen = p.data_format_train(source_name, 1, format_type="GEFG", meta=meta)
        try:

            while True:
                evid_mat, evid_len, evid_count, fact_mat, fact_len = next(data_gen)
                print(fact_len)

                last_time = time.time()
                state_seq, output_seq= sess.run(
                    [ops['state_seq'], ops['output_seq']],
                    feed_dict={ops['evid_mat']: evid_mat,
                               ops['evid_len']: evid_len,
                               ops['evid_count']: evid_count,
                               ops['fact_mat']: fact_mat,
                               ops['fact_len']: fact_len},
                    )

                cur_time = time.time()
                time_cost = cur_time - last_time
                total_cost = cur_time - start_time
                print(
                    '[INFO] 验证进行到第 %d 个文书，用时%.2f，总计用时%.2f' % (batch_count, time_cost, total_cost))
                # print('[INFO] Batch %d'%batch_count)
                # matplotlib 实现可视化loss
                t_fact = p.get_sentence(fact_mat)
                m_fact = p.get_sentence(output_seq)


                global_step += 1
        except StopIteration:

            print('[INFO] Validation Finished, Report has been written in file %s',)

train_GEFG()