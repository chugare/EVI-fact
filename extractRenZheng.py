# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/9/10 下午12:51
#   南京大学软件学院 Nanjing University Software Institute
#
import xml.etree.ElementTree as ET
import os
import re
import jieba
import math
import json
root_dic = '/Users/simengzhao/Desktop/项目/FactExtraction/2015'
filter_choose = ['证人证言','被告人供述和辩解']


def calc_evid_type():
    fl = os.listdir(root_dic)
    evids = {}
    count = 0
    for file in fl:
        print(file)
        if file.endswith('.xml'):

            print(root_dic + '/' + file)
            try:
                stree = ET.ElementTree(file=root_dic + '/' + file)

                renzheng = []
                for ele in stree.iter(tag='ZJJL'):
                    zl = next(ele.iterfind(path='ZJMX/ZL'))
                    zl = zl.attrib['value']
                    if zl not in evids:
                        evids[zl] = 0
                    evids[zl]+=1
            except StopIteration:
                pass
            count += 1
    res_file = open('evid_type_count.txt','w',encoding='utf-8')
    for k in evids:
        res_file.write('%s\t%d\n'%(k,evids[k]))

def extract_ZhengYan(root_dic):
    fl = os.listdir(root_dic)
    log = open('analyse_result.txt','w',encoding='utf-8')
    count = 0
    for file in fl:
        print(file)
        if file.endswith('.xml'):

            print(root_dic+'/'+file)
            try:
                stree = ET.ElementTree(file = root_dic+'/'+file)
                fact = next(stree.iter('RDSS')).attrib['value']

                renzheng = []
                for ele in stree.iter(tag='ZJJL'):
                    zl = next(ele.iterfind(path='ZJMX/ZL'))
                    if zl.attrib['value'] in filter_choose:
                        renzheng.append(ele.attrib['value'])

                if len(renzheng)>1:
                    log.write('<FACT%d>%s</FACT%d>\n' % (count, fact, count))
                    for rz in renzheng:
                        log.write('<EVID%d>%s</EVID%d>\n' % (count, rz, count))
            except StopIteration:
                pass
            count += 1

def sentence_simplified_process(str):
    str = re.sub('（[一二三四五六七八九十]*\d*）','',str)
    str = re.sub('\d*年\d月\d日','',str)

    return str




data_report()