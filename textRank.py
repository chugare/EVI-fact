import jieba
# import pkuseg
import json
import math
import numpy as np
import re
import Evaluate
from sklearn.metrics.pairwise import cosine_similarity
class extractSummary:
    def readJSONDoc_sen(self,doc_name,map_id= True):
        self.facts = []
        self.evids = []
        doc_f = open(doc_name,'r',encoding='utf-8')
        docs_js = json.load(doc_f)
        for doc_js in docs_js:
            fact = doc_js['fact']
            fact = jieba.lcut(fact)

            # self.facts.append(fact)
            evids = doc_js['evid']
            evid_sentences = []
            for e in evids:
                sen_tmp  =  re.split("[，。；]+",e)
                for s in sen_tmp:
                    s = s.strip()
                    if len(s)<=5:
                        continue
                    evid_sentences.append(jieba.lcut(s))
            if map_id:
                fact_id = []
                for f in fact:
                    if f  in self.GRAM2N:
                        fact_id.append(self.GRAM2N[f])
                evid_ids = []
                for evid in evid_sentences:
                    evid_id = []
                    for w in evid:
                        if w in self.GRAM2N:
                            evid_id.append(self.GRAM2N[w])
                    if len(evid_id)<1:
                        continue
                    evid_ids.append(evid_id)
                # self.evids += evids
                yield fact_id,evid_ids
            else:
                yield  fact,evid_sentences
    def readJSONDoc_para(self,doc_name):

        self.facts = []
        self.evids = []
        doc_f = open(doc_name,'r',encoding='utf-8')
        docs_js = json.load(doc_f)
        for doc_js in docs_js:

            yield doc_js['fact'],doc_js['evid']
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
        try:
            res = float(s_c)/(math.log(len(sen1))+math.log(len(sen2))+0.1)
        except ZeroDivisionError:
            print(sen1)
            print(sen2)


    def covage(self,doc_g):
        for fact,evids in doc_g:
            res = ''
            for e in evids:
                lead = str(e).split('，')[0]
                res+=lead

        yield fact,res
    def lead(self,doc_g):
        for fact,evids in doc_g:
            res = ''
            lead = str(evids[0]).split('，')[0]
            res+=lead

            yield fact,res
    def lexRank(self,doc_g):
        for fact,evids in doc_g:
            pass
    def Tfidf(self,doc_g):
        for fact,evid_sen in doc_g:
            t = Tf_idf()
            fact_v = t.tf_calc(fact)
            tf_v = []
            for sen in evid_sen:
                v = t.tf_calc(sen)
                sim = cosine_similarity([fact_v,v])[0][1]
                tf_v.append((''.join(sen),v,sim))
            tf_v.sort(key=lambda x:x[2],reverse=True)
            res = ''
            for s in tf_v:
                if len(s[0])+len(res)<100:
                    res = res + s[0]
                else:
                    break

            yield ''.join(fact),res

    def id2sen(self,ids):
        sen = ''
        for id in ids:
            sen += self.N2GRAM[id]
        return sen
    def textRank(self,doc_g):

        for fact, evids in doc_g:
            evids_seg = evids
            V = np.ones([len(evids_seg)], np.float32)
            E = np.zeros([len(evids_seg), len(evids_seg)], np.float32)

            d = 0.85
            for i in range(len(evids_seg) - 1):

                for j in range(i + 1, len(evids_seg)):
                    sim = self.simility(evids_seg[i], evids_seg[j])
                    E[i][j] = sim
                    E[j][i] = sim
            W_o = np.sum(E, 1)

            def V_iter():
                V_t = np.ones([len(evids_seg)], np.float32)
                for i in range(len(evids_seg)):
                    sum1 = 0.0
                    for j in range(len(evids_seg)):
                        sum1 += E[j][i] * V[j] / (W_o[j] + 0.001)
                    V_t[i] = (1 - d) + d * sum1
                return V_t

            def loss(V, V_t):
                loss_v = 0
                for i in range(len(V)):
                    loss_v += math.fabs(V[i] - V_t[i])
                return loss_v

            for i in range(100):
                V_t = V_iter()
                loss_v = loss(V, V_t)
                if loss_v < 1e-7:
                    V = V_t
                    break
                V = V_t
            V_r = V
            V = []
            print(len(evids))
            for i in range(len(evids)):
                V.append((i, V_r[i]))

            V.sort(key=lambda x: x[1], reverse=True)
            if len(V) > 10:
                lv = 10
            else:
                lv = len(V)

            res = ''
            for k in range(lv):
                ids, loss = V[k]
                res += es.id2sen(evids[ids])
            yield self.id2sen(fact),res


class Tf_idf:
    def __init__(self,dic=None,doc_file=None):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.idf = {}
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
            dic_file = open(dic,'r',encoding='utf-8')
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
    cosim_sum = 0.0
    cosim_c = 0
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
        cosim_sum += np.sum(np.array(cos_sim))
        cosim_c += len(cos_sim)
        prec+=llss
        print(llss)
    print("=============")
    print(cosim_sum/cosim_c)
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

def Eval_with_generator(gen):
    res_table  = []
    c = 0
    for ref_sen,gen_sen in gen:
        try:
            c+=1
            res = Evaluate.ROUGE_eval(ref_sen,gen_sen)
            res_table.append(res)
            print('[INFO] CASE %d finish evaluation'%c)
            print(res)
        except ValueError:
            print('[ERROR] CASE %d failed in  evaluation'%c)
            pass

    Evaluate.do_eval(res_table)

if __name__ == '__main__':
    es = extractSummary()
    # doc_g_id = es.readJSONDoc_sen('test_data.json')
    # doc_g_char = es.readJSONDoc_para('test_data.json')
    doc_d_sen = es.readJSONDoc_sen('test_data.json',False)
    # ldg = es.lead(doc_g=doc_g_char)
    # trg = es.textRank(doc_g_id)
    tfidf = es.Tfidf(doc_d_sen)
    Eval_with_generator(tfidf)


