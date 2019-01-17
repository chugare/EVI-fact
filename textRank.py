import jieba
import pkuseg
import json


seg = pkuseg.pkuseg()
fp = open('test_data.json','r',encoding='utf-8')
td = json.load(fp)
for l in td:
    words = seg.cut(l['fact'])
    print(l['fact'])
    print(words)


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
            # self.facts.append(fact)
            evids = docs_js['evid']
            evids =[seg.cut(e) for e in evids]
            # self.evids += evids
            yield fact,evids

    def read_dic(self):
        dic_file = open('_CHAR_DIC.txt', 'r', encoding='utf-8')
        for line in dic_file:
            word = line.split(' ')[0]
            index = int(line.split(' ')[1].strip())
            self.GRAM2N[word] = index
            self.RANK[word] = 1.0
            self.N2GRAM[index] = word
    def __init__(self,docs,dict=None):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.RANK = {}
        self.read_dic()
        self.readJSONDoc(docs)
    def calcRank(self,fact_seg,evids_seg):

        rank = {}
        contect = []
        for e in evids_seg:
            for w in e:
                if e not in rank:
                    rank[e] = 1.0

