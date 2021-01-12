from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import numpy as np

def dict_to_array(dict_log:dict)->list:

    ''' 函数功能:   归类后的数据被存储为dict格式 需要将其转换为list以制作数据集
                  创建空表，将每次读取到的序列追加进表内
        need:   numpy
        note:   用list append执行很快 np.concatenate慢十倍以上'''
    i = 0
    print_key = 10000
    
    for k,v in dict_log.items():
        
        data = np.array(v)
        
        
        try:
            dataset.append(data)
        except:
            dataset = []
            dataset.append(data)
        
        i+=1
        if (i%print_key)==0:
            print('already to array ',i,' areas.')

    return dataset

test = dict_to_array(log_np_convert)
  
