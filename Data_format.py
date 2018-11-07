# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/10/30 上午11:34
#   南京大学软件学院 Nanjing University Software Institute
#
import xml.etree.ElementTree as ET
import os
import re
import jieba
import math
import json
import random

filter_choose = ['证人证言','被告人供述和辩解']

def XML2JSON_extract(root_dic):
    fl = os.listdir(root_dic)
    data_file = open('RAW_DATA.json','w',encoding='utf-8')
    count = 0
    data_set = []
    fl = [root_dic+'/'+f for f in fl]
    for file in fl:
        if not file.endswith('.xml'):
            try:
                fl_a = os.listdir(file)
                fl_a = [file+'/'+f for f in fl_a]
                fl += fl_a
            except Exception:
                pass
        else:

            try:
                stree = ET.ElementTree(file = file)
                fact = next(stree.iter('RDSS')).attrib['value']

                renzheng = []
                for ele in stree.iter(tag='ZJJL'):
                    zl = next(ele.iterfind(path='ZJMX/ZL'))
                    if zl.attrib['value'] in filter_choose:
                        renzheng.append(ele.attrib['value'])

                if len(renzheng)>1:
                    res = {
                        'fact':fact,
                        'evid':renzheng
                    }
                    data_set.append(res)
                    # log.write('<FACT%d>%s</FACT%d>\n' % (count, fact, count))
                    # for rz in renzheng:
                    #     log.write('<EVID%d>%s</EVID%d>\n' % (count, rz, count))
            except StopIteration:
                pass
            count += 1
    print("[INFO] 人证信息提取完毕，总共提取文书%d篇"%count)
    json.dump(data_set,data_file,ensure_ascii=False,indent=2)


def read_json(name):
    source = open(name,'r',encoding='utf-8')
    dt = json.load(source)
    for i in dt:
        yield i

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

def sent_format(sentence):
    patterns = [
        r"\([一二三四五六七八九十]*\d*\)[，、．.,\s]*",
        r"（[一二三四五六七八九十]*\d*）[，、．.,\s]*",
        r"[一二三四五六七八九十]*\d*[，、．.,\s]+",
        r"[⑼]",
        "被告人的供述与辩解：*",
        "被告人.*?供述([与和]辩解)?：*",
        "证人证言及辨认笔录：*",
        "\A(证人)*.*?(证词|证言)?证实：*",
        "证人.*?证言：*",
        ".*?的证言(。|，)证实",
        "未到庭笔录证明",
        "辨认笔录及照片",
        "证明：",
        "经审理查明"
    ]
    for p in patterns:
        sentence = re.sub(p,'',sentence)
    return sentence

def data_format():
    max_len_evid = 800
    min_len_evid = 10
    max_len_fact = 600
    min_len_fact = 10
    max_count_evid = 50

    res_file = open('FORMAT_data.json','w',encoding='utf-8')
    dataset = []
    count = [0 for _ in range(6)]
    for set in read_json('RAW_DATA.json'):

        if len(set['fact'])>max_len_fact:
            count[0] +=1
            continue
        elif len(set['evid'])>max_count_evid:
            count[1] += 1
            continue
        elif len(max(set['evid'],key = lambda x:len(x))) > max_len_evid:
            count[2] += 1
            continue
        else:
            fact = sent_format(set['fact'])
            if len(fact) < min_len_fact:
                count[3] += 1
                continue
            evids = []
            for e in set['evid']:
                e = sent_format(e)
                if len(e)< min_len_evid:
                    continue
                else:
                    evids.append(e)
            if len(e) == 0 :
                count[4] += 1
                continue

            dataset.append(set)
            continue
    for i in count:
        print(i)
    json.dump(dataset,res_file, ensure_ascii=False, indent=2)

def seperate_data_set():
    source = open('FORMAT_data.json', 'r', encoding='utf-8')
    dt = json.load(source)
    random.shuffle(dt)
    train_file = open('train_data.json', 'w', encoding='utf-8')
    test_file = open('test_data.json', 'w', encoding='utf-8')
    json.dump(dt[:400],test_file,ensure_ascii=False, indent=2)
    json.dump(dt[400:],train_file,ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # root_dic = 'F:\\交通肇事罪文书\\故意杀人罪'
    root_dic = 'F:\\交通肇事罪文书\\故意杀人罪'
    # XML2JSON_extract(root_dic)


        # if len(i['fact'])<40:
        #     print('%d :"%s"'%(len(i['fact']),i['fact']))
    # seperate_data_set()