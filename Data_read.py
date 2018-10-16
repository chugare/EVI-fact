# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/29 下午1:25
#   南京大学软件学院 Nanjing University Software Institute
#

from  collections import OrderedDict
from pyexcel_xls import get_data
from pyexcel_xls import  save_data
import os
import jieba
def read_xls():
    data_base = '/Users/simengzhao/Desktop/数据/Evi-fact/'
    files = os.listdir(data_base)
    count = 1000
    for f in files:
        if count<=0:
            break
        msg = get_data(data_base+f)
        keys = msg.keys()
        print(keys)
        for k in keys:
            yield msg[k]
        count -= 1
class word_segmenter:
    def __init__(self,segment_file_name=None):
        self.sf_dic = self._read_stop()

    def _read_stop(self,file_name = None):
        sf_dic = {}
        sf = open(file_name or 'stopwords.txt','r',encoding='utf-8')
        for line in sf:
            sf_dic[line.strip()] = 0
        print('[INFO] Stop words reading process finished')
        return  sf_dic
    def cut_sentence(self,str):
        res = jieba.lcut(str)
        for r in res:
            if r in self.sf_dic:
                res.remove(r)
        return res