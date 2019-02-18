import json
import os
str_gefg_nnlm = """
rouge-1 f : 0.283107
rouge-1 p : 0.466776
rouge-1 r : 0.247308
rouge-2 f : 0.091257
rouge-2 p : 0.173276
rouge-2 r : 0.077695
rouge-l f : 0.145787
rouge-l p : 0.310457
rouge-l r : 0.143363
"""
str_lead = """
rouge-1 f : 0.236707
rouge-1 p : 0.525567
rouge-1 r : 0.174324
rouge-2 f : 0.104529
rouge-2 p : 0.264111
rouge-2 r : 0.076566
rouge-l f : 0.119834
rouge-l p : 0.377242
rouge-l r : 0.114837
"""

def divide_result(str1):
    ls = str1.split('\n')
    name = []
    res = []
    for l in ls:
        if len(l)<2:
            continue
        k = l.split(' ')
        name.append(k[0]+'-'+k[1])
        res.append(k[-1])
    print('\n'.join(name))
    print('\n'.join(res))

def read_from_gate_report():
    file = open('Gate_report.txt','r',encoding='utf-8')
    try:
        jsfile = open('gate_report_label.json','r',encoding='utf-8')
        Case = json.load(jsfile)
        jsfile.close()
    except Exception:
        Case = {}
    if len(Case)<1:
        Case = {
            'num':0,
            'case':[]
        }
    evid = []
    evid_w = []
    fact_w = []
    label = []
    labeled_record = Case['num']
    count = labeled_record
    for l in file:
        megs = l.split('\t')
        if len(megs)<7:
            evid.append(megs[1])
            evid_w.append(int(megs[0]))
        else:
            if len(fact_w)==0:
                fact_w = megs
            else:
                if labeled_record <= 0:
                    fact_s = megs
                    fact = ''.join(fact_s)
                    print("========证据部分如下,当前显示第%d个证据========"%(Case['num']))

                    for  i in range(len(evid)):
                        print('[%d|%d\t] %s'%(i,evid_w[i],evid[i]))

                    print("========事实部分如下========")
                    print('#\t'+fact)


                    inp = 999
                    while inp< 0 or inp>len(evid):
                        if inp == -1:
                            print("中断，退出")
                            return
                        print("输入最合适证据的编号")
                        try:
                            inp = int(input())
                        except ValueError:
                            inp = 0
                        label.append(int(inp))
                    inp = 999
                    while inp < 0 or inp > len(evid):
                        print("输入备选证据的编号")
                        try:
                            inp = int(input())
                        except ValueError:
                            inp = 0
                        label.append(int(inp))
                    inp = 999
                    while inp < 0 or inp > len(evid):
                        print("输入备选适证据的编号")
                        try:
                            inp = int(input())
                        except ValueError:
                            inp = 0
                        label.append(int(inp))
                    c = {
                        'fact':fact,
                        'evid':evid,
                        'evid_w':evid_w,
                        'label':label
                    }
                    Case['case'].append(c)
                    Case['num'] +=1
                    jsfile = open('gate_report_label.json', 'w', encoding='utf-8')
                    json.dump(Case,jsfile,ensure_ascii=False)
                    jsfile.close()
                else:
                    labeled_record -=1
                evid = []
                evid_w = []
                fact_w = []
                label = []

if __name__ == '__main__':
    read_from_gate_report()