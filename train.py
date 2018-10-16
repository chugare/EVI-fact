# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/8/19 下午10:22
#   南京大学软件学院 Nanjing University Software Institute
#
import tensorflow as tf
from model import  Model
class Train:
    def __init__(self):
        self.model =  Model()

    def run(self):
        t_ops = self.model.train();
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run()

