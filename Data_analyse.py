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


def idf_count(lib):
    fl = os.listdir(lib)
    count  = len(fl)
    word_dic = {}
    idf_file = open('idf.txt','w',encoding='utf-8')
    tmp_dic= {}
    for file in fl:
        print(file)
        if file.endswith('.xml'):
            stree = ET.ElementTree(file=root_dic + '/' + file)
            qw = next(stree.iter('QW')).attrib['value']
            words = jieba.lcut(qw)
            for w in words:
                if w not in tmp_dic:
                    tmp_dic[w] = ''
            for w in tmp_dic:
                if w not in word_dic:
                    word_dic[w] = 0
                word_dic[w] += 1

    for w in word_dic:
        if w ==' ':
            continue
        idf_file.write(w+' '+str(math.log(count/word_dic[w]+1))+'\n')

def read_idf():
    idf_file = open('idf.txt','r',encoding='utf-8')
    res = {}
    for line in idf_file:
        w = line.split(' ')
        res[w[0]] = float(w[1].strip())
    return res

def calc_tf(words):
    res = {}
    for w in words:
        if w not in res:
            res[w] = 0
        res[w] += 1

    return  res

def share_word(f_e_set,fname):
    idf = read_idf()
    fact = f_e_set['fact']
    evids = f_e_set['evid']
    fact = sentence_simplified_process(fact)
    fact_c = jieba.lcut(fact)
    fact_tf = calc_tf(fact_c)
    fact_res_file = open('fact_integrated/'+fname+'.txt','w',encoding='utf-8')
    fact_seg_file = open('fact_segmented/'+fname+'.txt','w',encoding='utf-8')
    fact_res_file.write('#  FACT : '+fact+'\n')
    for evid in evids:
        evid_w = jieba.lcut(sentence_simplified_process(evid))
        evid_tf = calc_tf(evid_w)
        score1 = 0.0    #   计算权重值的时候使用来自 事实 的tfidf值
        score2 = 0.0    #   计算权重值的时候使用来自 证据 的tfidf值
        for word in evid_w:
            for fword in fact_c:
                if word == fword:
                    idf_v = 3.1
                    if word  in idf:
                        idf_v = idf[word]

                    score1 +=  fact_tf[word]*idf_v
                    score2 +=  evid_tf[word]*idf_v
        fact_res_file.write('>  EVID : %f\t%f\t%s\n'%(score1,score2,evid))

    fact_segs = fact.split('。')
    for fact_seg in fact_segs:
        fact_c = jieba.lcut(fact_seg)
        fact_tf = calc_tf(fact_c)
        fact_seg_file.write('#  FACT SEG : ' + fact_seg + '\n')
        for evid in evids:
            evid_w = jieba.lcut(sentence_simplified_process(evid))
            evid_tf = calc_tf(evid_w)
            score1 = 0.0  # 计算权重值的时候使用来自 事实 的tfidf值
            score2 = 0.0  # 计算权重值的时候使用来自 证据 的tfidf值
            for word in evid_w:
                for fword in fact_c:
                    if word == fword:
                        idf_v = 3.1
                        if word in idf:
                            idf_v = idf[word]
                        score1 += fact_tf[word] * idf_v
                        score2 += evid_tf[word] * idf_v
            fact_seg_file.write('>  EVID : %f\t%f\t%s\n' % (score1, score2, evid))

def sentence_simplified_process(str):
    str = re.sub('（[一二三四五六七八九十]*\d*）','',str)
    str = re.sub('\d*年\d月\d日','',str)

    return str


