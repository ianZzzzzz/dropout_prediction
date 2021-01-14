'''
    代码功能：
        读取json文件    
        选用合适长度的样本
    待做：
        使用n-gram进行词计数
        使用td-idf分配词权重

        进入模型
'''
from typing import List, Dict
from numpy import ndarray
import numpy as np

TEST_OR_NOT = False
print_batch = int(1000000)
chunk_size = int(10000) # enable only when TEST_OR_NOT = True

def _t(function):
    from functools import wraps
    import time
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print ('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer

@_t
def read_or_write_json(
    path:str
    ,mode:str
    ,log = None):
    ''' 读写json文件
        mode控制 read or write
    '''
    import json

    def w(__log,__path):
        if type(__log[1])!=type([]):
            # json不支持ndarray
            # 用json导出 array 要先 .tolist() 读取的时候直接np.array()
            for i in range(len(_log)):
                __log[i] = __log[i].tolist()

        json.dump(__log,open(__path,'w'))
        return None

    def r(__log,__path)->list:
        _list = json.load(open(__path,'r'))
        return _list

    return eval(mode)(log,path)
def cut_toolong_tooshort(
    log_list: list
    ,up:int
    ,down:int
    )-> list:
    '''
       本函数根据设定的上下限 返回长度在上下限之间的序列构成的list
       
    '''

    uesful_series = []
    useless_series = []
    for series in log_list:
        length = len(series)
        
        if (length>down)and(length<up):
            uesful_series.append(series)
        else: 
            useless_series.append(series)
    
    return uesful_series
def find_avg_length_of_series(log_list: list)->list:
    len_array = np.zeros( len(log_list)+1,dtype = np.uint32)
    for i in range(len(log_list)):
        length =  len(log_list[i])
        len_array[i] = length

    series = len_array
    mean_ = np.mean(series)
    return mean_
def plot_histogram(log_list:list):
    '''
        描绘列表中序列长度分布的直方图 
    '''
    from matplotlib import pyplot as plt 
    import numpy as np
    plt.hist([len(s) for s in log_list], bins = 100) # 横坐标精度

    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

    ''' 细节版
        len_array = np.zeros( len(log_list)+1,dtype = np.uint32)
        for i in range(len(log_list)):
            length =  len(log_list[i])
            len_array[i] = length

        series = len_array

        min_ = int(np.min(series))
        max_ = int(np.max(series))
        gap =  max_ - min_

        bins_ = [
            min_
            ,(min_+int(0.10*gap))
            ,(min_+int(0.20*gap))
            ,(min_+int(0.30*gap))
            ,(min_+int(0.40*gap))
            ,(min_+int(0.50*gap))
            ,(min_+int(0.60*gap))
            ,(min_+int(0.70*gap))
            ,(min_+int(0.80*gap))
            ,(min_+int(0.90*gap))
            ,max_
            ]

        plt.hist( series, bins =  bins_)
        plt.show() '''

json_export_path = 'washed_log_list.json'

reader = read_or_write_json(
    path= json_export_path
    ,mode = 'r')
log_list = reader
useful_list = cut_toolong_tooshort(log_list,up = 2000,down = 100)
# use n-gram can use more useful data
plot_histogram(useful_list) 
avg_series_len = find_avg_length_of_series(useful_list)







