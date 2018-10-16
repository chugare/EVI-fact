# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/8/19 下午1:43
#   南京大学软件学院 Nanjing University Software Institute
#

ULSW = [' ','\t','\n']
import  numpy as np
import os
import sys
class OneHotEncoder:

    def __init__(self):
        self.dic = {}
        self.wordlist = []
        self.init_dic()
        self.read_dict()

    def read_dict(self):

        #从保存的dict文件中读取每一个单词对应的值，并声称一个dict类型的词典

        try:
            ldc_file = open('dict.txt','r',encoding='utf-8')
        except IOError:
            ldc_file = open('dict.txt','w',encoding='utf-8')
            ldc_file.close()
            return {}
        for line in ldc_file:
            turple = line.split(' ')
            self.dic[turple[0]] = int(turple[1])
            self.wordlist.append(turple[0])
        ldc_file.close()
        return self.dic

    def init_dic(self,EraseCurrentFile=False):

        #初始化词典，生成第一个表示文档结束的字符

        if not os.path.exists('dict.txt') or EraseCurrentFile:
            ldc_file = open('dict.txt','w',encoding='utf-8')
            ldc_file.write('EOD 0\n')
        else:
            pass

        #文档的结束表示用 EOD 表示
    def create_dic(self,fname):
        df = open(fname,'r',encoding='utf-8')
        self.init_dic(True)
        letter_dict = self.dic
        count = 1
        for line in df:
            for letter in line:
                if letter not in ULSW:
                    if letter not in letter_dict:
                        letter_dict[letter] = count
                        self.write_dict(letter, count)
                        count += 1
        df.close()
    def write_dict(self,name,index):

        #每一次更新字典的时候，都会向文档中添加一个新的单词
        ldc_file = open('dict.txt', 'a',encoding='utf-8')
        ldc_file.write(name+' '+str(index)+'\n');
        self.dic[name] = index
        self.wordlist.append(name)
        sys.stdout.write('%s is the  %d st character\n'%(name,index))
        ldc_file.close()
    def get_code(self,letter):
        return self.dic[letter]
    def get_word(self,arg):
        if hasattr(arg,'shape') and len(arg.shape) == 2:
            w = ''
            for wl in arg:
                num = np.argmax(wl)
                w += (self.wordlist[num])
            return w
        elif hasattr(arg,'shape') and len(arg.shape)==1:
            num = np.argmax(arg)
            return self.wordlist[num]
        else:
            w = self.wordlist[arg]
            return w
    def one_hot_single(self,line,isNumber = False):
        line = line.replace(' ','')
        count = len(self.dic)
        letter_dict = self.dic
        store = []
        for letter in line:
            if letter not in ULSW:
                if letter not in letter_dict:
                    letter_dict[letter] = count
                    self.write_dict(letter, count)
                    count += 1
                if isNumber:
                    store.append(letter_dict[letter])
                else:
                    one_hot_vec = np.zeros(VEC_SIZE,np.int32)
                    one_hot_vec[letter_dict[letter]] = 1
                    store.append(one_hot_vec)

        return np.array(store)

    def one_hot_queue_G(self,file):

        #从文档中读取句子并且生成onehot编码序列，此函数作为一个generator存在
        #每次生成一行单词的onehot编码序列，最后生成的数据类型是numpy的array


        infile = open(name=file, mode='r')
        letter_dict = read_dict()
        count = len(letter_dict)

        for line in infile:
            store = []
            line = line.decode('utf-8')
            for letter in line:
                if letter not in ULSW:
                    if letter not in letter_dict:
                        letter_dict[letter] = count
                        write_dict(letter, count)
                        count += 1
                        one_hot_vec = np.zeros(VEC_SIZE,np.int32)
                        one_hot_vec[letter_dict[letter]] = 1
                        store.append(one_hot_vec)
            store.append(np.zeros(VEC_SIZE,np.int32))
            arr = np.array(store)
            yield arr
