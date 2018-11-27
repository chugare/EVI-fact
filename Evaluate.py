# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/11/1 下午9:26
#   南京大学软件学院 Nanjing University Software Institute
#
from rouge import Rouge
import json
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


if __name__ == '__main__':
    do_eval('ABS_VALID_valid.json')

