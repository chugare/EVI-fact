# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/11/1 下午9:26
#   南京大学软件学院 Nanjing University Software Institute
#
from rouge import Rouge
import json
import preprocess
import  sys
def ROUGE_eval(standard_sen,generated_sen):
    r = Rouge()
    s = ''
    g = ''
    for i in standard_sen:
        s += i+' '

    for i in generated_sen:
        g += i + ' '
    res = r.get_scores(g,s)
    return res
def ROUGE_eval_(standard_sen,generated_sen):
    r = Rouge()
    res = r.get_scores(generated_sen,standard_sen)
    return res

def do_eval(fname):
    jsfile = open(fname,'r',encoding='utf-8')

    res = json.load(jsfile)
    sum = res[0][0]
    c = 0
    for r in res:
        if c == 0 :
            c+= 1
            continue
        r = r[0]
        for t in r:
            for i in r[t]:
                sum[t][i] += r[t][i]
        c+=1
    avg = {}
    for t in sum:
        avg[t] = {}
        for r in sum[t]:
            avg[t][r] = float(sum[t][r])/c
            print("%s %s : %f"%(t,r,avg[t][r]))
def gate_value_report_write(fname,evids_ids,fact_ids,gate_v):
    p = preprocess.Preprocessor(False)
    fact = p.get_sentence(fact_ids)
    evids = []
    e_w = []
    for e in evids_ids:
        if e[0] == 2:
            e_w.append(0)
            for i in range(len(e[0])):
                if e[i] == 1:
                    e = e[:i]
                    break
            evids.append(p.get_sentence(e))
        else:
            break
    f = open(fname,'a',encoding='utf-8')

    for g in gate_v:
        e_w[g] +=1
    for i in range(len(evids)):

        f.write('%d\t%s'%(e_w[i],evids[i]))
        f.write('\n')
    for g in gate_v:
        f.write('%d\t'%g)
    f.write('\n')
    for f_c in fact:
        f.write(f_c+'\t')
    f.write('\n')

    f.close()


if __name__ == '__main__':
    do_eval(sys.argv[1])

