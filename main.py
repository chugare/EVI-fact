# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/9/4 下午2:20
#   南京大学软件学院 Nanjing University Software Institute
#
import sys
import preprocess
import model
import Evaluate
import tensorflow as tf
import os
import time
import json
import logging
import datetime



class log_train:
    def __init__(self, name, t=None):
        if not os.path.exists(name):
            os.mkdir(name)
        if t == None:
            t = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        name = "./%s/%s_"%(name,name)+t
        self.filename = name
        self.log_file = open(name,'w',encoding='utf-8')
    def write_log(self,data_list):
        data_list = [str(i) for i in data_list]
        self.log_file.write('\t'.join(data_list)+'\n')
    def write_exception(self,e):
        self.log_file.write(e)
    def flush(self):
        self.log_file.close()
        self.log_file = open(self.filename,'a',encoding='utf-8')
    def close(self):
        self.log_file.close()
class log_eval:
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
    p = preprocess.Preprocessor(SEG_BY_WORD=meta['seg_by_word'])
    logger = log_train(meta['name']) # meta
    data_meta = meta['data_meta'] # meta

    gate_ana_res_file = open('Gate_report.txt','w',encoding='utf-8')
    gate_ana_res_file.close()
    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = meta['model']() # meta
    if 'model_meta' in meta:
        m.set_meta(meta['model_meta'])
    ops = m.build_model('train')

    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True

    )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 训练配置，包括参数初始化以及读取检查点
        init = tf.global_variables_initializer()
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess.graph.finalize()
        train_writer = tf.summary.FileWriter(summary_dir,sess.graph)
        start_epoch = 0
        global_step = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
            global_step += int(checkpoint.split('-')[-2])
        start_time = time.time()

        # 开始训练
        for i in range(start_epoch,epoch):
            data_gen = p.data_provider(source_name, meta=data_meta)
            try:
                batch_count = 0
                while True:
                    try:
                        last_time = time.time()

                        train_res = m.train_fun(sess,data_gen,ops,i)

                        loss = train_res['loss']
                        merge = train_res['merge']
                        if 'acc' in train_res:
                            acc = train_res['acc']
                        else:
                            acc = 0
                        cur_time =time.time()
                        time_cost = cur_time-last_time
                        total_cost = cur_time-start_time
                        if global_step % 1 == 0:
                            train_writer.add_summary(merge,global_step)
                            # logger.write_log([global_step/10,loss,total_cost])
                        print('[INFO] Batch %d 训练结果：LOSS=%.2f ACCURACY:%.2f 用时: %.2f 共计用时 %.2f' % (batch_count, loss,acc,time_cost,total_cost))

                        # print('[INFO] Batch %d'%batch_count)
                        # matplotlib 实现可视化loss
                        batch_count += 1
                        global_step += 1
                    except StopIteration:
                        print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                        saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
                        break
                    except Exception as e:
                        logging.exception(e)
                        print("[INFO] 因为程序错误停止训练，开始保存模型")
                        saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
            except StopIteration:
                print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)

            except KeyboardInterrupt:
                print("[INFO] 强行停止训练，开始保存模型")
                saver.save(sess, os.path.join(checkpoint_dir, meta['name']+'_summary-'+str(global_step)), global_step=i)
                break

def valid_protype(meta):
    # 设置训练配置内容
    source_name = meta['eval_data'] # meta
    checkpoint_dir = os.path.abspath(meta['checkpoint_dir']) # meta
    summary_dir = os.path.abspath(meta['summary_dir']) # meta
    p = preprocess.Preprocessor(SEG_BY_WORD=meta['seg_by_word'])
    # logger = log_train(meta['name']) # meta
    data_meta = meta['data_meta'] # meta

    # 模型搭建
    # with tf.device('/cpu:0'):
    # with tf.device('/device:GPU:0'):
    m = meta['model']() # meta
    ops = m.build_model('valid')

    # 训练过程
    saver = tf.train.Saver()
    config = tf.ConfigProto(
        # log_device_placement=True
        # report_tensor_allocations_upon_oom=True
    )
    with tf.Session(config=config) as sess:
        # 配置，包括参数初始化以及读取检查点
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        sess.graph.finalize()
        if checkpoint:
            saver.restore(sess, checkpoint)
            print('[INFO] 从检查点 %s 进行验证'%checkpoint)
        else:
            print('[ERROR] 指定的验证文档不存在，验证程序退出')
            return
        start_time = time.time()

        # 开始验证
        global_step = 0
        data_gen = p.data_provider(source_name, meta=data_meta)
        report_data = {
            'G_R1': 0.0,
            'G_R2': 0.0,
            'G_RL': 0.0,
        }
        res_list = []

        report_txt = open('rt.txt', 'w', encoding='utf-8')

        try:
            while True:
                try:
                    last_time = time.time()
                    inter_res = m.inter_fun(sess,data_gen,ops)
                    out_seq = inter_res['out_seq']
                    fact_seq = inter_res['fact_seq']

                    if source_name.endswith('WV'):
                        out_sen = p.wordvec.get_sentence(out_seq)
                        fact_sen = fact_seq
                    else:
                        fact_sen = p.get_sentence(fact_seq)
                        out_sen = p.get_sentence(out_seq,len(fact_sen))
                    if len(out_sen) < len(fact_sen):
                        out_sen = out_sen[:len(fact_sen)]

                    cur_time = time.time()
                    time_cost = cur_time-last_time
                    total_cost = cur_time-start_time


                    print('[INFO] 第 %d 个测试例子验证结束  用时: %.2f 共计用时 %.2f 得到生成队列：' % (global_step,time_cost,total_cost))
                    print('TRUE:'+fact_sen)
                    print('GEN:'+out_sen)
                    rouge_v = Evaluate.ROUGE_eval(fact_sen,out_sen)
                    report_txt.write('> 正确文本%d\n' % global_step)
                    report_txt.write('> %s\n> \n' % fact_sen)
                    report_txt.write('> 生成文本%d\n' % global_step)
                    report_txt.write('> %s\n\n' % out_sen)

                    res_list.append(
                        rouge_v
                    )
                    global_step += 1

                    # if global_step>10:
                    #     break
                except StopIteration:

                    report_file = open(meta['name']+'_valid.json','w',encoding='utf-8')
                    json.dump(res_list,report_file)
                    print("[INFO] 验证结束，正在生成报告..." )
                    break
                except Exception as e:
                    logging.exception(e)
        except KeyboardInterrupt:
            print("[INFO] 强行停止验证")


# train_ABS()
# valid_GEFG()
# train_GEFG()

GEFG_WV = model.GEFG_WV()
ABS = model.ABS_model()
S2S = model.SEQ2SEQ()

GEFG_WV_meta={
    'name':'GEFG_WV',
    'seg_by_word': True,
    'train_data':'train_data.json',
    'checkpoint_dir':'checkpoint_GEFG',
    'summary_dir':'summary_GEFG',
    'model':model.GEFG_WV,
    'data_meta':{
        'NAME':'SE_WV',
        'MEL':GEFG_WV.MAX_EVID_LEN,
        'MEC':GEFG_WV.MAX_EVIDS,
        'MFL':GEFG_WV.MAX_FACT_LEN,
        'VEC_SIZE':300,
        'BATCH_SIZE':1
    }
}

ABS_meta={
    'name':'ABS',
    'seg_by_word':False,
    'train_data':'train_data.json',
    'checkpoint_dir':'checkpoint_ABS',
    'summary_dir':'summary_ABS',
    'model':model.ABS_model,
    'data_meta':{
        'NAME':'ABS',
        'C':ABS.C,
        'V':ABS.V,
        'BATCH_SIZE':ABS.BATCH_SIZE
    }

}
GEFG_WV_VALID_meta={
    'name':'GEFG_WV',
    'seg_by_word': False,
    'eval_data':'test_data.json',
    'checkpoint_dir':'checkpoint_GEFG',
    'summary_dir':'summary_GEFG',
    'model':model.GEFG_WV,
    'data_meta':{
        'NAME':'SE',
        'MEL':GEFG_WV.MAX_EVID_LEN,
        'MEC':GEFG_WV.MAX_EVIDS,
        'MFL':GEFG_WV.MAX_FACT_LEN,
        'BATCH_SIZE':1
    }
}
ABS_VALID_meta ={
    'name':'ABS_VALID',
    'eval_data':'test_data.json',
    'checkpoint_dir': 'checkpoint_ABS',
    'summary_dir':'summary_ABS_V',
    'seg_by_word':False,
    'model':model.ABS_model,
    'data_meta':{
        'NAME':'ABS_infer',
        'C':ABS.C,
        'V':ABS.V,
        'BATCH_SIZE':ABS.BATCH_SIZE
    }
}


S2S_meta={
    'name':'S2S',
    'seg_by_word': False,
    'train_data':'train_data.json',
    'checkpoint_dir':'checkpoint_S2S',
    'summary_dir':'summary_S2S',
    'model':model.SEQ2SEQ,
    'data_meta':{
        'NAME':'CE',
        'MEL':S2S.MAX_EVID_LEN,
        'MFL':S2S.MAX_FACT_LEN,
        'BATCH_SIZE':1
    }
}
S2S_VALID_meta={
    'name':'S2S',
    'seg_by_word': False,
    'eval_data':'test_data.json',
    'checkpoint_dir':'checkpoint_S2S',
    'summary_dir':'summary_S2S',
    'model':model.SEQ2SEQ,
    'data_meta':{
        'NAME':'CE',
        'MEL':S2S.MAX_EVID_LEN,
        'MFL':S2S.MAX_FACT_LEN,
        'BATCH_SIZE':1
    }
}

if __name__ == '__main__':
    print(sys.argv)
    if sys.argv[1] == 'GEFG':
        if sys.argv[2] == 'v':
            valid_protype(meta=GEFG_VALID_meta)
        if sys.argv[2] == 't':
            train_protype(meta= GEFG_meta)
    elif sys.argv[1] == 'ABS':
        if sys.argv[2] == 'v':
            valid_protype(meta=ABS_VALID_meta)
        if sys.argv[2] == 't':
            train_protype(meta= ABS_meta)
    elif sys.argv[1] == 'S2S':
        if sys.argv[2] == 'v':
            valid_protype(meta=S2S_VALID_meta)
        if sys.argv[2] == 't':
            train_protype(meta= S2S_meta)

