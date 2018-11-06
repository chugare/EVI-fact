# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/10/30 上午11:34
#   南京大学软件学院 Nanjing University Software Institute
#
import os
import re
import jieba
import math
import json
import random

def read_file():
    rf = open('analyse_result.txt','r',encoding='utf-8')
    res = {'fact':'',
    'evid':[],}
    for line in rf:
        fact_pattern = '<FACT(.*?)>(.*?)</FACT.*>'
        evid_pattern = '<EVID(.*?)>(.*?)<EVID.*>'
        f = re.findall(fact_pattern,line)
        e = re.findall(evid_pattern,line)
        if len(f) >0 :
            if res['fact']!= '':
                yield res
            res = {'fact': f[0][1],
                   'evid': [], }

        elif len(e) >0:
            res['evid'].append(e[0][1])
        else:
            pass

    yield res

def read_json(name):
    source = open(name,'r',encoding='utf-8')
    dt = json.load(source)
    for i in dt:
        yield i

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

#calc_evid_type()
def data_report():
    evid_count = 0
    evid_count_max = 0
    evid_count_min = 999
    evid_len_sum = 0
    evid_len_max = 0
    evid_len_min = 99999
    fact_len_sum = 0
    fact_len_max = 0
    fact_len_min = 99999
    count = 0
    evid_length_dis = [0 for i in range(101)]
    evid_count_dis = [0 for i in range(101)]
    fact_length_dis = [0 for i in range(101)]
    for set in read_json():
        evid_count_max = max([evid_count_max,len(set['evid'])])
        evid_count_min = min([evid_count_min,len(set['evid'])])
        evid_count += len(set['evid'])
        for e in set['evid']:
            ind = (int)(len(e)/40)
            if ind <100:
                evid_length_dis[ind] += 1
            else:
                evid_length_dis[100] += 1
            evid_len_max = max([evid_len_max,len(e)])
            evid_len_min = min([evid_len_min,len(e)])

            evid_len_sum += len(e)
        ind = (int)(len(set['fact'])/20)
        if ind < 100:
            fact_length_dis[ind] += 1
        else:
            fact_length_dis[100] +=1
        ind = len(set['evid'])
        if ind <100:
            evid_count_dis[ind] += 1
        else:
            evid_count_dis[100] += 1
        fact_len_sum += len(set['fact'])
        fact_len_max  = max(len(set['fact']),fact_len_max)
        fact_len_min = min(fact_len_min,len(set['fact']))
        count += 1
    main_report = """
    evid_count = %d
    evid_count_max = %d
    evid_count_min = %d
    evid_avg_len = %d
    evid_len_max = %d
    evid_len_min = %d
    fact_avg_len = %d
    fact_len_max = %d
    fact_len_min = %d
    count = %d"""%(
    evid_count,
    evid_count_max ,
    evid_count_min ,
    evid_len_sum/evid_count ,
    evid_len_max ,
    evid_len_min ,
    fact_len_sum/count ,
    fact_len_max ,
    fact_len_min ,
    count
    )

    log_file = open('data_report.txt','w')
    # log_file.write(main_report)
    # log_file.write('\n')
    for c in evid_length_dis:
        log_file.write('%d\t'%c)
    log_file.write('\n')

    for c in evid_count_dis:
        log_file.write('%d\t'%c)
    log_file.write('\n')

    for c in fact_length_dis:
        log_file.write('%d\t'%c)
    log_file.write('\n')

def data_reformat():
    max_len_evid = 800
    max_len_fact = 600
    max_count_evid = 50
    res_file = open('data.json','w',encoding='utf-8')
    dataset = []
    for set in read_file():
        if len(set['fact'])>max_len_fact:
            continue
        elif len(set['evid'])>max_count_evid:
            continue
        elif len(max(set['evid'],key = lambda x:len(x))) > max_len_evid:
            continue
        else:
            dataset.append(set)
            continue
    json.dump(dataset,res_file, ensure_ascii=False, indent=2)

def seperate_data_set():
    source = open('data.json', 'r', encoding='utf-8')
    dt = json.load(source)
    random.shuffle(dt)
    train_file = open('train_data.json', 'w', encoding='utf-8')
    test_file = open('test_data.json', 'w', encoding='utf-8')
    json.dump(dt[:400],test_file,ensure_ascii=False, indent=2)
    json.dump(dt[400:],train_file,ensure_ascii=False, indent=2)

if __name__ == '__main__':

    seperate_data_set()