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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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


class Tf_idf:
    def __init__(self,dic=None,doc_file=None):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.idf = {}
        dic_file = open(dic,'r',encoding='utf-8')
        try:
            _data_file = open('_tfidf_meta.json','r',encoding='utf-8')
            _data_t = json.load(_data_file)
            self.GRAM2N = _data_t['G']
            self.N2GRAM = _data_t['N']
            self.idf = _data_t['I']
        except Exception:
            if dic is None or doc_file is None:
                print('[ERROR] Require data file to initialize')
                return
            for line in dic_file:
                wd = line.split(' ')
                self.GRAM2N[wd[0]] = int(wd[1].strip())
                self.N2GRAM[int(wd[1].strip())] = wd[0]
                self.idf[wd[0]] = 0.0
            ga = Tf_idf.read_doc_all(doc_file)
            self.idf_calc(ga)
            _data_file = open('_tfidf_meta.json','w',encoding='utf-8')
            obj = {
                'G':self.GRAM2N,
                'N':self.N2GRAM,
                'I':self.idf
            }
            json.dump(obj,_data_file,ensure_ascii=False)

    def idf_calc(self,doc_gen):
        # doc_data = json.load(self.doc_file)
        doc_num = 0.0
        print('[INFO] Start calc idf')
        for doc in doc_gen:
            tmp_idf = {}
            doc_num+=1
            for w in doc:
                if w not in tmp_idf:
                    tmp_idf[w] = 1

            for w in tmp_idf:
                if w in self.idf:
                    self.idf[w] += 1
            if int(doc_num) % 100 == 0:
                print('[INFO] %d of doc read'%doc_num)
        print('[INFO] All docs have been read')
        for w in self.idf:
            self.idf[w] = math.log(doc_num/(self.idf[w]+1))
        print('[INFO] All idf value have been calculated')
    def tf_calc(self,sen):
        tf = np.zeros(shape=[len(self.N2GRAM)])
        tf_idf = np.zeros(shape=[len(self.N2GRAM)])
        l = len(sen)
        for word in sen:
            if word in self.GRAM2N:
                tf[self.GRAM2N[word]] = (tf[self.GRAM2N[word]]+1)
                tf_idf[self.GRAM2N[word]] = tf[self.GRAM2N[word]]*self.idf[word]/l

        return  tf_idf
    @staticmethod
    def read_doc_all(fname):
        file_all = open(fname,'r',encoding='utf-8')
        data_all = json.load(file_all)
        for d in data_all:
            fact = d['fact']
            evids = d['evid']
            fact = jieba.cut(fact)
            yield fact
            for e in  evids:
                e = jieba.cut(e)
                yield e
    @staticmethod
    def read_doc_case(fname):
        file_all = open(fname,'r',encoding='utf-8')
        data_all = json.load(file_all)

def t_generate_tfidf():
    t = Tf_idf('_WORD_DIC.txt','FORMAT_data.json')
    gdt = json.load(open('gate_report_label.json','r',encoding='utf-8'))
    case = gdt['case']
    num = gdt['num']
    prec = 0
    for c in case:
        m = c['evid_w'][0]
        im = 0
        for i in range(len(c['evid_w'])):
            if m< c['evid_w'][i]:
                m = c['evid_w'][i]
                im = i
        evid_w = [int(e) for e in c['evid_w']]
        f = c['fact']
        label = c['label']

        fact_vec = t.tf_calc(jieba.lcut(f))
        cos_sim = []
        for i  in range(len(c['evid'])):
            e = c['evid'][i]
            e_vec = t.tf_calc(jieba.lcut(e))
            sim = cosine_similarity([fact_vec,e_vec])
            sim = sim[0][1]
            cos_sim.append(sim)
            # print("%s  %f"%(c['evid_w'][i],sim))
        # s1 = []
        # for i in range(len(cos_sim)):
        #     s1.append([i,cos_sim[i]])
        # s1.sort(key=lambda x:x[1])


        # print("=========case=========")
        im_cos_sim = cos_sim.index(max(cos_sim))
        # dif = calc_seq_diff(cos_sim,evid_w)
        llss = evid_w[im_cos_sim]/float(len(f))
        prec+=llss
        print(llss)
    print(prec/num)

def calc_seq_diff(seq1,seq2):
    s1 = []
    s2 = []
    l = len(seq1)
    for i in range(l):
        s1.append((i,seq1[i]))
    s1.sort(key=lambda x:x[1])
    for i in range(l):
        s2.append((i,seq2[i]))
    s2.sort(key=lambda x:x[1])


    ss1 = {s1[i][0]:i for i in range(l)}
    d_sum = 0.0
    for i in range(l):
        d_index =abs(ss1[i] - s2[i][0])/float(l*l)*2
        d_sum += d_index
    return d_sum

# def idf_count(lib):
#     fl = os.listdir(lib)
#     count  = len(fl)
#     word_dic = {}
#     idf_file = open('idf.txt','w',encoding='utf-8')
#     tmp_dic= {}
#     for file in fl:
#         print(file)
#         if file.endswith('.xml'):
#             stree = ET.ElementTree(file=root_dic + '/' + file)
#             qw = next(stree.iter('QW')).attrib['value']
#             words = jieba.lcut(qw)
#             for w in words:
#                 if w not in tmp_dic:
#                     tmp_dic[w] = ''
#             for w in tmp_dic:
#                 if w not in word_dic:
#                     word_dic[w] = 0
#                 word_dic[w] += 1
#
#     for w in word_dic:
#         if w ==' ':
#             continue
#         idf_file.write(w+' '+str(math.log(count/word_dic[w]+1))+'\n')
#
# def read_idf():
#     idf_file = open('idf.txt','r',encoding='utf-8')
#     res = {}
#     for line in idf_file:
#         w = line.split(' ')
#         res[w[0]] = float(w[1].strip())
#     return res
#
# def calc_tf(words):
#     res = {}
#     for w in words:
#         if w not in res:
#             res[w] = 0
#         res[w] += 1
#
#     return  res
#
# def share_word(f_e_set,fname):
#     idf = read_idf()
#     fact = f_e_set['fact']
#     evids = f_e_set['evid']
#     fact = sentence_simplified_process(fact)
#     fact_c = jieba.lcut(fact)
#     fact_tf = calc_tf(fact_c)
#     fact_res_file = open('fact_integrated/'+fname+'.txt','w',encoding='utf-8')
#     fact_seg_file = open('fact_segmented/'+fname+'.txt','w',encoding='utf-8')
#     fact_res_file.write('#  FACT : '+fact+'\n')
#     for evid in evids:
#         evid_w = jieba.lcut(sentence_simplified_process(evid))
#         evid_tf = calc_tf(evid_w)
#         score1 = 0.0    #   计算权重值的时候使用来自 事实 的tfidf值
#         score2 = 0.0    #   计算权重值的时候使用来自 证据 的tfidf值
#         for word in evid_w:
#             for fword in fact_c:
#                 if word == fword:
#                     idf_v = 3.1
#                     if word  in idf:
#                         idf_v = idf[word]
#
#                     score1 +=  fact_tf[word]*idf_v
#                     score2 +=  evid_tf[word]*idf_v
#         fact_res_file.write('>  EVID : %f\t%f\t%s\n'%(score1,score2,evid))
#
#     fact_segs = fact.split('。')
#     for fact_seg in fact_segs:
#         fact_c = jieba.lcut(fact_seg)
#         fact_tf = calc_tf(fact_c)
#         fact_seg_file.write('#  FACT SEG : ' + fact_seg + '\n')
#         for evid in evids:
#             evid_w = jieba.lcut(sentence_simplified_process(evid))
#             evid_tf = calc_tf(evid_w)
#             score1 = 0.0  # 计算权重值的时候使用来自 事实 的tfidf值
#             score2 = 0.0  # 计算权重值的时候使用来自 证据 的tfidf值
#             for word in evid_w:
#                 for fword in fact_c:
#                     if word == fword:
#                         idf_v = 3.1
#                         if word in idf:
#                             idf_v = idf[word]
#                         score1 += fact_tf[word] * idf_v
#                         score2 += evid_tf[word] * idf_v
#             fact_seg_file.write('>  EVID : %f\t%f\t%s\n' % (score1, score2, evid))

def sentence_simplified_process(str):
    str = re.sub('（[一二三四五六七八九十]*\d*）','',str)
    str = re.sub('\d*年\d月\d日','',str)

    return str
if __name__ == '__main__':
    t_generate_tfidf()

