# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/11/6 下午3:27
#   南京大学软件学院 Nanjing University Software Institute
#

# main.py
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


