# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/29 下午3:26
#   南京大学软件学院 Nanjing University Software Institute
#
import jieba
from Data_read import read_xls,word_segmenter


def compare_str(wl1,wl2):
    pos = []
    w1c = 0
    same_count = 0
    for w1 in wl1:
        w2c = 0
        index = []
        for w2 in wl2:
            if w1==w2:
                index.append(w2c)
                same_count+=1
            w2c +=1
        pos.append(index)
        w1c+=1
    return pos

def calc_share_word():
    turple_file = open('turple.txt', 'w')
    tmp_f = open('tmp_fact.txt','w')
    tmp_e = open('tmp_evi.txt','w')
    dt_g = read_xls()
    row_count = -1
    ws = word_segmenter()
    evidences = []
    facts = []
    for dt in dt_g:
        for line in dt:
            row_count += 1
            if row_count == 0 :
                for evidence in  line[1:]:
                    tmp_e.write(evidence+'\n')
                    evidences.append(ws.cut_sentence(evidence))
                continue
            try:
                tmp_f.write(line[0]+'\n')
                fact = ws.cut_sentence(line[0])
                facts.append(fact)
                colomn_count = 0
                for col in line[1:]:
                    if col == 1 or col == '1':
                        wl2 = evidences[colomn_count]
                        wl1 = fact
                        turple_file.write(' '.join(wl1)+'###')
                        turple_file.write(' '.join(wl2)+'\n')
                        pos = compare_str(wl1,wl2)
                        share_word = []
                        for p in range(len(pos)) :
                            if len(pos[p]) >0:
                                share_word.append(fact[p])
                        print(share_word)
                    colomn_count+=1
            except IndexError:
                continue


calc_share_word()
