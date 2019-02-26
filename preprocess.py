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
# import pkuseg
import sys
import os
import struct
from sklearn.metrics.pairwise import cosine_similarity
class WORD_VEC:
    def __init__(self):
        self.vec_dic = {}
        self.word_list = []
        self.vec_list = []
        self.num = 0
        print('[INFO] Start load word vector')
        self.read_vec()
        self.seg = pkuseg.PKUSeg()
    def dump_file(self):
        file = open('word_vec.char','w',encoding='utf-8')
        file.write(str(self.num)+' 300\n')
        for w in self.vec_dic:
            vec_f = [str(i) for i in self.vec_dic[w]]
            vec_str = ' '.join(vec_f)
            file.write(w+' '+vec_str+'\n')
        file.close()
    @staticmethod
    def ulw(word):
        pattern = [
            r'[,.\(\)（），。\-\+\*/\\_|]{2,}',
            r'\d+',
            r'[qwertyuiopasdfghjklzxcvbnm]+',
            r'[ｑｗｅｒｔｙｕｉｏｐａｓｄｆｇｈｊｋｌｚｘｃｖｂｎｍ]+',
            r'[QWERTYUIOPASDFGHJKLZXCVBNM]+',
            r'[ＱＷＥＲＴＹＵＩＯＰＡＳＤＦＧＨＪＫＬＺＸＣＶＢＮＭ]+',
            r'[ⓐ ⓑ ⓒ ⓓ ⓔ ⓕ ⓖ ⓗ ⓘ ⓙ ⓚ ⓛ ⓜ ⓝ ⓞ ⓟ ⓠ ⓡ ⓢ ⓣ ⓤ ⓥ ⓦ ⓧ ⓨ ⓩ]+',
            r'[Ⓐ Ⓑ Ⓒ Ⓓ Ⓔ Ⓕ Ⓖ Ⓗ Ⓘ Ⓙ Ⓚ Ⓛ Ⓜ Ⓝ Ⓞ Ⓟ Ⓠ Ⓡ Ⓢ Ⓣ Ⓤ Ⓥ Ⓦ Ⓧ Ⓨ Ⓩ ]+',
        ]
        ulwf = open('uslw.txt','a',encoding='utf-8')
        for p in pattern:
            mr = re.match(p,word)
            if mr is not None:
                ulwf.write(word+'\n')
                ulwf.close()
                return True
        return  False
    @staticmethod
    def clear_ulw(filename):
        vec_file = open(filename,'r',encoding='utf-8')
        meg = next(vec_file).split(' ')
        num = int(meg[0])
        file = open('word_vec.char', 'w', encoding='utf-8')

        count = 0

        for l in vec_file:
            m = l.strip().split(' ')
            w = m[0]
            if WORD_VEC.ulw(w):
                continue
            count+=1
            if count%10000 == 0:
                p = float(count)/num*100
                sys.stdout.write('\r[INFO] write cleared vec data, %d finished'%count)
            file.write(l)
                # vec_dic[w] = vec
        print('\n Final count : %d'%count)
    def read_vec(self):
        path = os.path.abspath('.')
        print(path)
        # path = '/'.join(path.split('\\')[:-1])+'/sgns.merge.char'
        # path = 'F:/python/word_vec/sgns.merge.char'
        # path = 'D:\\赵斯蒙\\EVI-fact\\word_vec.char'
        path = 'word_vec.char'
        vec_file = open(path,'r',encoding='utf-8')
        # meg = next(vec_file).split(' ')
        # num = int(meg[0])
        # self.num = num
        count = 0
        for l in vec_file:

            m = l.strip().split(' ')
            w = m[0]
            vec = m[1:]
            vec =[float(v) for v in m[1:]]
            # if WORD_VEC.ulw(w):
            #     continue
            count+=1
            if count%10000 == 0:
                sys.stdout.write('\r[INFO] Load vec data, %d finished'%count)
            if count == 500000:
                break
            self.vec_list.append(vec)
            self.word_list.append(w)
            self.vec_dic[w] = np.array(vec,dtype=np.float32)

        print('\n[INFO] Vec data loaded')
        self.num = count
    def get_min_word(self,word):
        vec = self.vec_dic[word]
        dis = cosine_similarity(self.vec_list,[vec])
        dis = np.reshape(dis,[-1])
        dis_pair = [(i,dis[i]) for i in range(len(dis))]
        dis_pair.sort(key= lambda x:x[1],reverse=True)
        for i in range(10):
            print(self.word_list[dis_pair[i][0]])

    def get_min_word_v(self, vec):
        dis = cosine_similarity(self.vec_list, [vec])
        dis = np.reshape(dis, [-1])
        i = np.argmax(dis)
        return self.word_list[i]
    def get_sentence(self,vlist,l):
        result = ''
        x = 0
        for vec in vlist:
            if x == l:
                break
            print('[INFO] Search for nearest word on index %d'%x)
            dis = cosine_similarity(self.vec_list, [vec])
            dis = np.reshape(dis, [-1])
            i = np.argmax(dis)
            x+= 1
            print(self.word_list[i])
            result += self.word_list[i]

        return result
    def sen2vec(self,sen):
        sen = self.seg.cut(sen)
        vec_out = []
        for w in sen:
            if w in self.vec_dic:
                vec_out.append(self.vec_dic[w])
        return vec_out
    def case_to_vec(self,raw_file):
        f = open(raw_file,'r',encoding='utf-8')
        cases = json.load(f)
        fmax = 0
        fmin = 999

        emax = 0
        emin = 999
        fjson = open('case_wordvec.json','w',encoding='utf-8')
        clist = []
        count = 0
        for case in cases:
            f = case['fact']
            es = case['evid']
            fv = self.sen2vec(f)
            if len(fv)>fmax :
                fmax = len(fv)
            elif len(fv)<fmin:
                fmin = len(fv)
            evs = []
            for e in es:
                ev = self.sen2vec(e)
                if len(ev) > emax:
                    emax = len(ev)
                elif len(fv) < emin:
                    emin = len(ev)
                evs.append(ev)
            c = {'fact':fv,
                 'evid':evs}
            clist.append(c)
            count+=1
            if count%100 == 0:
                print('[INFO] Case to json, %d cases finished'%count)

            if count == 10:
                break
        json.dump(clist,fjson)

class Preprocessor:
    def __init__(self,SEG_BY_WORD = True):
        self.SEG_BY_WORD = SEG_BY_WORD
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.freq_threshold = 0
        self.read_dic()
        self.wordvec = None
        self.ULSW = ['\n', '\t',' ','\n']

    def read_dic(self):
        try:
            if self.SEG_BY_WORD:
                dic_file = open('_WORD_DIC.txt', 'r', encoding='utf-8')
            else:
                dic_file = open('_CHAR_DIC.txt', 'r', encoding='utf-8')
            for line in dic_file:
                word = line.split(' ')[0]
                index = int(line.split(' ')[1].strip())
                self.GRAM2N[word] = index
                self.N2GRAM[index] = word
        except FileNotFoundError:
            print('[INFO] 未发现对应的*_DIC.txt文件，需要先初始化，初始化完毕之后重新运行程序即可')


    def init_dic(self, source_file):
        #   第一次建立字典的时候调用
        dic_count = {}
        seg = PKUSeg()
        try:
            data_gen = self.read_file(source_file)
            count = 0
            for aj in data_gen:
                if count%10 == 0:
                    sys.stdout.write("\r[INFO] %d of aj has been read to mem.."%count)
                if self.SEG_BY_WORD:
                    grams = seg.cut(aj['fact'])
                else:
                    grams = aj['fact'].strip()
                for gram in grams:
                    if gram not in dic_count:
                        dic_count[gram] = 0
                    dic_count[gram] += 1
                evids = aj['evid']
                for e in evids:
                    if self.SEG_BY_WORD:
                        grams = seg.cut(e)
                    else:
                        grams = e.strip()
                    for gram in grams:
                        if gram not in dic_count:
                            dic_count[gram] = 0
                        dic_count[gram] += 1
                count += 1
            self.GRAM2N['<u>'] = 0
            self.GRAM2N['<e>'] = 1
            self.GRAM2N['<s>'] = 2
            index = 3
            print('\n[INFO] File read successfully, now drop word less than %d'%self.freq_threshold)
            count_t = 0
            for word in dic_count:
                if count_t % 1000 == 0:
                    sys.stdout.write("\r[INFO] %f finished .." % (float(count_t) / len(dic_count)))
                if dic_count[word] >= self.freq_threshold:
                    if word not in self.ULSW:
                        self.GRAM2N[word] = index
                    index += 1
            print('\n[INFO] Dictionary built successfully')
        except FileNotFoundError:
            print("[ERROR] Source file \'%s\' not found" % (source_file))
        
        if self.SEG_BY_WORD:
            dic_file = open('_WORD_DIC.txt', 'w', encoding='utf-8')
        else:
            dic_file = open('_CHAR_DIC.txt', 'w', encoding='utf-8')
        count = 0
        for i in self.GRAM2N:
            if count%1000 == 0:
                sys.stdout.write("\r[INFO] %f of word has been written.."%(float(count)/index))
            count+=1
            dic_file.write('%s %d\n' % (i, self.GRAM2N[i]))

    def get_sentence(self, index_arr,cut_size = None):

        res = ''
        for i in range(len(index_arr)):
            if cut_size != None:
                if index_arr[i] > 1:
                    res+=(self.N2GRAM[index_arr[i]])
                if len(res)>cut_size:
                    break
            else:
                if index_arr[i] != 1:
                   res+=(self.N2GRAM[index_arr[i]])
                else:
                    break

        return res
    def get_char_list(self, index_arr):
        res = []
        for i in range(len(index_arr)):
            if index_arr[i] != 1:
               res.append(self.N2GRAM[index_arr[i]])
            else:
                break

        return res

    def Nencoder(self, ec_str):
        if self.SEG_BY_WORD:
            grams = jieba.lcut(ec_str)
        else:
            grams = ec_str
        ec_vecs = [2]

        for gram in grams:
            if gram in self.GRAM2N:
                ec_vecs.append(self.GRAM2N[gram])
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
    @staticmethod
    def context(title, pos, C):
        res = np.zeros([C], np.int32)
        for i in range(C):
            if pos-i-1<0:
                res[C - i - 1] = 0
            else:
                res[C - i - 1] = title[pos - i - 1]
        return res

    def data_provider(self,data_source,meta):

        #   输入的数据是原始的文本形式，在这个函数中进行查找，oh化并按batch划分
        #   由于是训练用的数据，所以会分批处理，输入的参数包括批次，context长度等信息
        #   结果输入由一个batchsize的list组成，每一个单元包括原始的bow编码，输出文本的上下文信息，以及下一个输出文本标签
        res_gen = self.read_file(data_source)
        format_type = meta['NAME']
        #   从文本源中得到基本格式的数据，类型为
        #     {
        #         evids:['',''],
        #         fact:'',
        #     }
        #   证据的文本分为多行
        # 根据所用模型的不同，输出数据的格式要求也有所不同，格式分为下面几种
        if format_type.startswith('ABS'):
            count = 0
            try:
                V = meta['V']
                C = meta['C']
                batch_size = meta['BATCH_SIZE']
            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V ,BATCH)')
                return
            art_vecs = []
            y_vecs = []
            yc_vecs = []
            for res in res_gen:
                evids = res['evid']
                title = res['fact']
                article = ''
                for e in evids:
                    article += e
                art_vec = self.bowencoder(self.Nencoder(article),V)
                title_vec = self.Nencoder(title)
                if format_type == 'ABS_infer':
                    yield art_vec,title_vec
                    continue
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

        elif format_type =='CE':
            # Concatenate Evidence
            count = 0
            try:
                mel = meta['MEL']
                mfl = meta['MFL']
                pass

            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
            for res in res_gen:
                evids = res['evid']
                fact = res['fact']
                c_evid = ''
                for e in evids:
                    c_evid += e


                evid_oh = self.Nencoder(c_evid)
                evid_len = len(evid_oh)
                if evid_len>mel:
                    continue
                tmp_vec = np.array(evid_oh)
                padded_vec = np.concatenate((tmp_vec,np.zeros([mel-len(tmp_vec)])))

                fact_vec = self.Nencoder(fact)
                fact_len = len(fact_vec)

                try:
                    fact_vec = np.concatenate([fact_vec,np.zeros([mfl-len(fact_vec)],dtype=np.int32)])
                except ValueError:
                    print("v error fact:")
                    print(fact_len)
                    continue
                yield padded_vec,evid_len,fact_vec,fact_len,fact
        elif format_type =='SE':
            # Separated Evidence
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
                    if i < len(evids):
                        evid_oh = self.Nencoder(evids[i])
                        tmp_vec = np.array(evid_oh)
                        try:
                            padded_vec = np.concatenate((tmp_vec,np.zeros([mel-len(tmp_vec)])))
                        except ValueError:
                            print("v error evid:")
                            print(len(tmp_vec))
                            continue
                        evid_vecs.append(padded_vec)
                        evid_lens.append(len(evid_oh))
                    else:
                        evid_vecs.append(np.zeros(mel))
                        evid_lens.append(1)

                fact_vec = self.Nencoder(fact)
                fact_len = len(fact_vec)

                try:
                    fact_vec = np.concatenate([fact_vec,np.zeros([mfl-len(fact_vec)],dtype=np.int32)])
                except ValueError:
                    print("v error fact:")
                    print(fact_len)
                    continue
                yield np.matrix(evid_vecs),np.array(evid_lens),len(evids),fact_vec,fact_len
        elif format_type =='SE_WV':
            # Separated Evidence with Word Vector
            # 将原始的证据分离输入，并且使用词向量编码
            # 输出格式为 一个数组表示所有的证据文本的词向量矩阵，一个向量表示事实文本的词向量矩阵
            count = 0
            if self.wordvec is None:
                wordvec =  WORD_VEC()
                self.wordvec = wordvec
            try:
                mel = meta['MEL']
                mec = meta['MEC']
                mfl = meta['MFL']
                vec_size = meta['VEC_SIZE']
                pass

            except KeyError:
                print('[ERROR] The meta data expected for data preparation required more info (require: C,V )')
                return
            for res in res_gen:
                evids = res['evid']
                fact = res['fact']
                evid_vecs = []
                evid_lens = []
                for i in range(mec):
                    if i < len(evids):
                        evid_vec = wordvec.sen2vec(evids[i])
                        tmp_vec = np.matrix(evid_vec)
                        try:
                            padded_vec = np.concatenate((tmp_vec,np.zeros([mel-len(tmp_vec),vec_size])))
                        except ValueError:
                            print("v error evid:")
                            print(len(tmp_vec))
                            continue
                        evid_vecs.append(padded_vec)
                        evid_lens.append(len(evid_vec))
                    else:
                        evid_vecs.append(np.zeros([mel,vec_size]))
                        evid_lens.append(0)

                fact_vec = wordvec.sen2vec(fact)
                fact_len = len(fact_vec)

                try:
                    fact_vec = np.concatenate([fact_vec,np.zeros([mfl-len(fact_vec),vec_size],dtype=np.int32)])
                except ValueError:
                    print("v error fact:")
                    print(fact_len)
                    continue
                yield evid_vecs,np.array(evid_lens),len(evids),fact_vec,fact_len,fact
        else:
            print("[ERROR] Declaration of format type is required")
def init():
    print('[INFO] 初始化字典/词典')
    p = Preprocessor(True)
    p.init_dic('RAW_DATA.json')
if __name__ == '__main__':


    jsfile = open('FORMAT_data.json','r',encoding='utf-8')
    data = json.load(jsfile)
    ellist = []
    for d in data:
        evid = d['evid']
        c = 0
        for e in evid:
            c += len(e)
        ellist.append(c)
    ellist.sort()
    all = len(ellist)
    for i in range(10):
        print(ellist[int(all*i/10)])
    print(ellist[-1])
    # wv.dump_file()
    # WORD_VEC.clear_ulw('F:/python/word_vec/sgns.merge.char')