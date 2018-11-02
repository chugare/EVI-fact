# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/30 下午6:53
#   南京大学软件学院 Nanjing University Software Institute
#
#   这个文件用于生成训练所需要的数据，主要包括以下几个方法
#   1. 从文件中读取，生成原始文本与摘要文本的序列
#   2. 将文本序列表示成字典序列
#   3. 生成batch数据
#   4. 去除停用词
import jieba
import numpy as np
import re
import json

class Preprocessor:
    def __init__(self):

        self.DIC = {}

        self.wordlist = {}

        self.freq_threshold = 20

        self.read_dic()
        self.ULSW = ['\n', '\t']

    def read_dic(self):
        try:
            dic_file = open('_dic.txt', 'r', encoding='utf-8')
            for line in dic_file:
                word = line.split(' ')[0]
                index = line.split(' ')[1]
                self.DIC[word] = index
                self.wordlist[index] = word
        except FileNotFoundError:
            pass

    def init_dic(self, source_file):
        #   第一次建立字典的时候调用
        dic_count = {}
        try:
            data_gen = self.read_file(source_file)
            for aj in data_gen:
                words = jieba.lcut(aj['fact'])

                for word in words:
                    if word not in dic_count:
                        dic_count[word] = 0
                    dic_count[word] += 1
                evids = aj['evid']
                for e in evids:
                    words = jieba.lcut(e)
                    for word in words:
                        if word not in dic_count:
                            dic_count[word] = 0
                        dic_count[word] += 1

            self.DIC['<unk>'] = 0
            self.DIC['<eos>'] = 1
            self.DIC['<sos>'] = 2

            index = 3
            for word in dic_count:
                if dic_count[word] >= self.freq_threshold:
                    if word not in self.ULSW:
                        self.DIC[word] = index
                    index += 1
            print('[INFO] Dictionary built successfully')
        except FileNotFoundError:
            print("[ERROR] Source file \'%s\' not found" % (source_file))
        dic_file = open('_dic.txt', 'w', encoding='utf-8')

        for i in self.DIC:
            dic_file.write('%s %d\n' % (i, self.DIC[i]))

    def get_sentence(self, index_arr):
        res = []
        for i in range(index_arr):
            if index_arr[i] != 1:
               res.append(self.wordlist[index_arr[i]])
            else:
                break

        return res


    def ohencoder(self, ec_str):
        words = jieba.lcut(ec_str)
        ec_vecs = [2]

        for word in words:
            if word in self.DIC:
                ec_vecs.append(self.DIC[word])
            else:
                # 当词典中没有对应的词时，简单的把单词变成unk符号，抑或是进行进一步的分词？
                ec_vecs.append(0)
        ec_vecs.append(1)
        return np.array(ec_vecs, np.int32)

    def bowencoder(self, ohcode, V):
        res = np.zeros([V], np.int32)
        for c in ohcode:
            res[c] = 1
        return res

    def read_file(self,data_source):
        source = open(data_source, 'r', encoding='utf-8')
        dt = json.load(source)
        for i in dt:
            yield i
    # def read_file(self, data_source):
    #     rf = open(data_source, 'r', encoding='utf-8')
    #     res = {'fact': '',
    #            'evid': [], }
    #     for line in rf:
    #         fact_pattern = '<FACT(.*?)>(.*?)</FACT.*>'
    #         evid_pattern = '<EVID(.*?)>(.*?)<EVID.*>'
    #         f = re.findall(fact_pattern, line)
    #         e = re.findall(evid_pattern, line)
    #         if len(f) > 0:
    #             if res['fact'] != '':
    #                 yield res
    #             res = {'fact': f[0][1],
    #                    'evid': [], }
    #
    #         elif len(e) > 0:
    #             res['evid'].append(e[0][1])
    #         else:
    #             pass
    #
    #     yield res

    #   生成context的函数
    @staticmethod
    def context(title, pos, C):
        res = np.zeros([C], np.int32)
        for i in range(C):
            res[C - i - 1] = title[pos - i - 1]
        return res

    def data_format_train(self, data_source, batch_size, format_type, meta):

        #   输入的数据是原始的文本形式，在这个函数中进行查找，oh化并按batch划分
        #   由于是训练用的数据，所以会分批处理，输入的参数包括批次，context长度等信息
        #   结果输入由一个batchsize的list组成，每一个单元包括原始的bow编码，输出文本的上下文信息，以及下一个输出文本标签
        res_gen = self.read_file(data_source)

        #   从文本源中得到基本格式的数据，类型为
        #     {
        #         evids:['',''],
        #         fact:'',
        #     }
        #   证据的文本分为多行
        # 根据所用模型的不同，输出数据的格式要求也有所不同，格式分为下面几种
        if format_type =='ABS':
            count = 0
            try:
                V = meta['V']
                C = meta['C']
            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
            art_vecs = []
            y_vecs = []
            yc_vecs = []
            for res in res_gen:
                evids = res['evid']
                title = res['fact']
                article = ''
                for e in evids:
                    article += e
                art_vec = self.bowencoder(self.ohencoder(article),V)
                title_vec = self.ohencoder(title)

                for p in range(len(title_vec)):
                    art_vecs.append(art_vec)
                    y_vecs.append(title_vec[p])
                    yc_vecs.append(Preprocessor.context(title_vec, p, C))
                    count += 1
                    if count % batch_size == 0:
                        yield art_vecs, yc_vecs, y_vecs
                        art_vecs = []
                        y_vecs = []
                        yc_vecs = []
        elif format_type =='Seg_OH':

            # 将原始的证据分离输入，并且使用one-hot编码方式
            # 输出格式为 一个数组表示所有的证据文本的one-hot编码，一个向量表示事实文本的one-hot编码
            count = 0
            try:
                pass

            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
            art_vecs = []
            y_vecs = []
            yc_vecs = []
            for res in res_gen:
                evids = res['evid']
                title = res['fact']
                art_vec = self.ohencoder()
                title_vec = self.ohencoder(title)

                for p in range(len(title_vec)):
                    art_vecs.append(art_vec)
                    y_vecs.append(title_vec[p])
                    yc_vecs.append(Preprocessor.context(title_vec, p, C))
                    count += 1
                    if count % batch_size == 0:
                        yield art_vecs, yc_vecs, y_vecs
                        art_vecs = []
                        y_vecs = []
                        yc_vecs = []
        elif format_type =='Int_OH':
            pass
            count = 0
            try:
                V = meta['V']
                C = meta['C']
            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
            art_vecs = []
            y_vecs = []
            for res in res_gen:
                evids = res['evid']
                title = res['fact']
                article = ''
                for e in evids:
                    article += e
                art_vec = self.ohencoder(article)
                title_vec = self.ohencoder(title)

                for p in range(len(title_vec)):
                    art_vecs.append(art_vec)
                    y_vecs.append(title_vec[p])
                    count += 1
                    if count % batch_size == 0:
                        yield art_vecs, y_vecs
                        art_vecs = []
                        y_vecs = []
        elif format_type =='GEFG':

            # 将原始的证据分离输入，并且使用one-hot编码方式
            # 输出格式为 一个数组表示所有的证据文本的one-hot编码，一个向量表示事实文本的one-hot编码
            count = 0
            try:
                mel = meta['MEL']
                mec = meta['MEC']
                mfl = meta['MFL']
                pass

            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
            for res in res_gen:
                evids = res['evid']
                fact = res['fact']
                evid_vecs = []
                evid_lens = []
                for i in range(mec):
                    if i<len(evids):
                        evid_oh = self.ohencoder(evids[i])
                        tmp_vec = np.array(evid_oh)
                        padded_vec = np.concatenate((tmp_vec,np.zeros([mel-len(tmp_vec)])))
                        evid_vecs.append(padded_vec)
                        evid_lens.append(len(evid_oh))
                    else:
                        evid_vecs.append(np.zeros(mel))
                        evid_lens.append(0)

                fact_vec = self.ohencoder(fact)
                fact_len = len(fact_vec)
                fact_vec = np.concatenate([fact_vec,np.zeros([mfl-len(fact_vec)],dtype=np.int32)])
                yield np.matrix(evid_vecs),np.array(evid_lens),len(evids),fact_vec,fact_len

        else:
            print("[ERROR] Declaration of format type is required")
    def data_format_eval(self, data_source,format_type, meta):
        #   输入的数据是原始的文本形式，在这个函数中进行查找，oh化并按batch划分
        #   验证的时候使用的是生成模型，所以不应该进行批处理，由模型自己产生的数据进行下一步推算，计算context的工作交由生成步骤来做
        res_gen = self.read_file(data_source)
        if format_type =='ABS':
            #输出原始语料的词袋模型信息，以及摘要文档的oh编码序列
            try:
                V = meta['V']
            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')

            for res in res_gen:
                evids = res['evid']
                title = res['fact']
                article = ''
                for e in evids:
                    article += e
                art_vec = self.bowencoder(self.ohencoder(article),V)
                title_vec = self.ohencoder(title)

                yield art_vec,title_vec
def init():
    p = Preprocessor()
    p.init_dic('tmp.txt')
if __name__ == 'main':
    init()