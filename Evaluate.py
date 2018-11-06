# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/11/1 下午9:26
#   南京大学软件学院 Nanjing University Software Institute
#
from rouge import  Rouge
def ROUGE_eval(standard_sen,generated_sen):
    r = Rouge()
    res = r.get_scores(generated_sen,standard_sen)
    return res
