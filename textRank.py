import jieba
import pkuseg
import json
import math
import numpy as np
import re



class textRank:
    def readJSONDoc(self,doc_name):
        seg = pkuseg.pkuseg()
        self.facts = []
        self.evids = []
        doc_f = open(doc_name,'r',encoding='utf-8')
        docs_js = json.load(doc_f)
        for doc_js in docs_js:
            fact = doc_js['fact']
            fact = seg.cut(fact)
            fact_id = []
            for f in fact:
                if f  in self.GRAM2N:
                    fact_id.append(self.GRAM2N[f])
            # self.facts.append(fact)
            evids = doc_js['evid']
            evid_sentences = []
            for e in evids:
                sen_tmp  =  re.split("[，。；]+",e)
                for s in sen_tmp:
                    s = s.strip()
                    if len(s)<=5:
                        continue
                    evid_sentences.append(seg.cut(s))
            evid_ids = []
            for evid in evid_sentences:
                evid_id = []
                for w in evid:
                    if w in self.GRAM2N:
                        evid_id.append(self.GRAM2N[w])
                evid_ids.append(evid_id)
            # self.evids += evids
            yield fact_id,evid_ids

    def read_dic(self):
        dic_file = open('_WORD_DIC.txt', 'r', encoding='utf-8')
        for line in dic_file:
            word = line.split(' ')[0]
            index = int(line.split(' ')[1].strip())
            self.GRAM2N[word] = index
            self.RANK[word] = 1.0
            self.N2GRAM[index] = word
    def __init__(self):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.RANK = {}
        self.read_dic()
    @staticmethod
    def simility(sen1,sen2):
        s_c = 0
        for w1 in sen1:
            for w2 in sen2:
                if w1 == w2:
                    s_c += 1
        return float(s_c)/(math.log(len(sen1))+math.log(len(sen2)))

    def id2sen(self,ids):
        sen = ''
        for id in ids:
            sen += self.N2GRAM[id]
        return sen
    def calcRank(self,evids_seg):

        V = np.ones([len(evids_seg)],np.float32)
        E = np.zeros([len(evids_seg),len(evids_seg)],np.float32)

        d = 0.85
        for i in range(len(evids_seg)-1):

            for j in range(i+1,len(evids_seg)):
                sim = self.simility(evids_seg[i],evids_seg[j])
                E[i][j] = sim
                E[j][i] = sim
        W_o = np.sum(E, 1)
        def V_iter():
            V_t = np.ones([len(evids_seg)],np.float32)
            for i in range(len(evids_seg)):
                sum1 = 0.0
                for j in range(len(evids_seg)):
                    sum1 += E[j][i]*V[j]/(W_o[j]+0.001)
                V_t[i] = (1-d) + d*sum1
            return V_t
        def loss(V,V_t):
            loss_v = 0
            for i in range(len(V)):
                loss_v += math.fabs(V[i]-V_t[i])
            return loss_v
        for i in range(100):
            V_t = V_iter()
            loss_v = loss(V,V_t)
            if loss_v<1e-7:
                break
            V = V_t
        return V

if __name__ == '__main__':
    tr = textRank()
    doc_g = tr.readJSONDoc('train_data.json')
    for fact,evids in doc_g:
        V_r = tr.calcRank(evids_seg=evids)
        V = []
        for i in range(len(evids)):
            V.append([i, V_r[i]])

        V = V.sort(key=lambda x:x[1])

