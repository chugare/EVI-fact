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

import datetime
class log_train:
    def __init__(self, name, t=None):
        if not os.path.exists(name):
            os.mkdir(name)
        if t == None:
            t = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        name = "./%s/%s_"%(name,name)+t

        self.log_file = open(name,'w',encoding='utf-8')
    def write_log(self,data_list):
        data_list = [str(i) for i in data_list]
        self.log_file.write('\t'.join(data_list)+'\n')


ERROR_LOG = open('error_log.txt','w',encoding='utf-8')


def train_protype(meta):
    # 设置训练配置内容
    epoch = 50
    source_name = meta['train_data'] # meta
    checkpoint_dir = os.path.abspath(meta['checkpoint_dir']) # meta
    summary_dir = os.path.abspath(meta['summary_dir']) # meta
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    p = preprocess.Preprocessor()
    logger = log_train(meta['name']) # meta
    data_meta = meta['data_meta'] # meta

    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = meta['model']() # meta
    ops = m.build_model('train')


    # 配置数据生成器的元数据
    # meta = {
    #     'MEL':m.MAX_EVID_LEN,
    #     'MEC':m.MAX_EVIDS,
    #     'MFL':m.MAX_FACT_LEN
    # }


    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True
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
            data_gen = p.data_format_train(source_name, 1, format_type=meta['name'], meta=data_meta)
            try:
                batch_count = 0
                while True:
                    try:
                        last_time = time.time()

                        train_fun = meta['train_fun'] # meta

                        loss,merge = train_fun(sess,data_gen,ops)
                        cur_time =time.time()
                        time_cost = cur_time-last_time
                        total_cost = cur_time-start_time
                        if global_step % 10 == 0:
                            train_writer.add_summary(merge,global_step/10)
                            logger.write_log([global_step/10,loss,total_cost])
                        print('[INFO] Batch %d 训练结果：NLL=%.6f  用时: %.2f 共计用时 %.2f' % (batch_count, loss ,time_cost,total_cost))

                        # print('[INFO] Batch %d'%batch_count)
                        # matplotlib 实现可视化loss
                        batch_count += 1
                        global_step += 1
                    except StopIteration:
                        print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                        saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary'), global_step=i)
                        break
                    # except Exception as e:
                    #
                    #     print("[INFO] 因为程序错误停止训练，开始保存模型")
                    #     saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary'), global_step=i)
            except StopIteration:
                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary'), global_step=i)

            except KeyboardInterrupt:
                print("[INFO] 强行停止训练，开始保存模型")
                saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary'), global_step=i)

def train_ABS():
    epoche = 50
    source_name = 'train_data.json'
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
        start_time = time.time()

        try:
            for i in range(epoche):
                data_g = processor.data_format_train(source_name, m.BATCH_SIZE, format_type='ABS', meta=meta)

                try:
                    batch_count = 0
                    while True:
                        last_time = time.time()
                        in_x,yc,y = next(data_g)
                        _,nll,gx = sess.run([ops['train_op'],ops['nll'],ops['gx']], feed_dict={ops['in_x']:in_x,ops['in_y']:y,ops['cont_y']:yc})


                        cur_time = time.time()
                        time_cost = cur_time - last_time
                        total_cost = cur_time - start_time
                        print('[INFO] Batch %d 训练结果：NLL=%.6f  用时: %.2f 共计用时 %.2f' % (
                        batch_count, nll, time_cost, total_cost))
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
    logger = log_train('GEFG')


    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = model.gated_evidence_fact_generation()
    ops = m.build_model('train')

    # 配置数据生成器的元数据
    # meta = {
    #     'MEL':m.MAX_EVID_LEN,
    #     'MEC':m.MAX_EVIDS,
    #     'MFL':m.MAX_FACT_LEN
    # }
    t_op = ops['train_op']

    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True
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
                    try:
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
                            logger.write_log([global_step/10,nll,total_cost])
                        print('[INFO] Batch %d 训练结果：NLL=%.6f  用时: %.2f 共计用时 %.2f' % (batch_count, nll,time_cost,total_cost))

                        # print('[INFO] Batch %d'%batch_count)
                        # matplotlib 实现可视化loss
                        batch_count += 1
                        global_step += 1
                    except StopIteration:
                        print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                        saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)
                        break
                    except Exception as e:
                        for i in fact_mat:
                            ERROR_LOG.write(str(i)+' ')
                        ERROR_LOG.write('\n')
                        print("[INFO] 因为程序错误停止训练，开始保存模型")
                        saver.save(sess, os.path.join(checkpoint_dir, 'GEFG_summary'), global_step=i)
            except StopIteration:
                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
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
                    '[INFO] 验证进行到第 %d 个文书，用时%.2f，总计用时%.2f' % (global_step, time_cost, total_cost))
                # print('[INFO] Batch %d'%batch_count)
                # matplotlib 实现可视化loss
                t_fact = p.get_sentence(list(fact_mat))
                m_fact = p.get_sentence(output_seq)


                global_step += 1
        except StopIteration:

            print('[INFO] Validation Finished, Report has been written in file %s',)
# train_ABS()
# valid_GEFG()
# train_GEFG()

GEFG = model.gated_evidence_fact_generation()
GEFG_meta={
    'name':'GEFG',
    'train_data':'train_data.json',
    'checkpoint_dir':'checkpoint_GEFG',
    'summary_dir':'summary_GEFG',
    'model':model.gated_evidence_fact_generation,
    'data_meta':{
        'MEL':GEFG.MAX_EVID_LEN,
        'MEC':GEFG.MAX_EVIDS,
        'MFL':GEFG.MAX_FACT_LEN
    },
    'train_fun':model.gated_evidence_fact_generation.train_fun,

}
ABS = model.ABS_model()
ABS_meta={
    'name':'ABS',
    'train_data':'train_data.json',
    'checkpoint_dir':'checkpoint_ABS',
    'summary_dir':'summary_ABS',
    'model':model.ABS_model,
    'data_meta':{
        'C':ABS.C,
        'V':ABS.V
    },
    'train_fun':model.ABS_model,

}

train_protype(meta= ABS_meta)