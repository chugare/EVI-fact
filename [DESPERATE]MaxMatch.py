# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/30 上午8:57
#   南京大学软件学院 Nanjing University Software Institute
#
STOPLETTER = [',','.','、','。','，','\n','\t']
def match(str1,str2):
    matchCut = {}
    rp = 0
    while rp < len(str1):

        rp_2 = 0
        while rp_2 < len(str2):
            rl = 0


            while  str1[rp+rl] == str2[rp_2+rl] and str1[rp+rl] not in STOPLETTER:
                rl+=1
                if rl > len(str2) - 1 - rp_2 or rl > len(str1) - 1 - rp:
                    break
            if rl >= 2:
                word = str1[rp:rp + rl]
                if word  not in matchCut:
                    matchCut[word] = 0
                matchCut[word] += 1
                rp_2 += rl
            else :
                rp_2 += 1
        rp += 1
    return matchCut

def match_statistic():
    f = open('tmp_fact.txt','r',encoding='utf-8')
    log = open('log.txt','w',encoding='utf-8')

    facts = []
    global_match_c = {}
    for line in f:
        facts.append(line)
    tc = len(facts)
    rc = 0
    for f1 in facts[:-1]:
        for f2 in facts[rc+1:]:
            m = match(f1,f2)
            for k in m:
                if k  not in global_match_c:
                    global_match_c [k] = 0
                global_match_c[k] += m[k]
        rc += 1
        print('[INFO] %.2f lines finished'%(float(rc)/tc*100))
    for k in global_match_c:
        log.write(k+' : '+str(global_match_c[k])+'\n')

match_statistic()