'''
    程序功能：制作数据标签
        
        使用convert函数将原始日志转换成以注册号为索引的字典
        存储字典为json
        按照参数切分样本和标签
        打包样本和标签存储为json
'''

TEST_OR_NOT = False
print_batch = int(1000000)
read_chunk_size  = int(10000) # enable only when TEST_OR_NOT = True

label_rate = int(50) # 50 mean's 50%
PAD_LENGTH = 10

def _import_():
    # preprocess of "log file to time series"
    import pandas as pd
    # import cudf as pd # nvidia GPU only # !pip install cudf 
    from typing import List, Dict
    from numpy import ndarray
    from numpy import datetime64
    from pandas import DataFrame
    import numpy as np
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
@_t
def word_counter(log_list:list)-> Dict[str,dict]:
    '''
        计算单词频次
    '''
    from matplotlib import pyplot as plt 
    import numpy as np

    action_type_counter = { '1':0,  '2':0, '3':0, '4':0 }
    action_counter = {
        11:0,12:0,13:0,14:0,15:0       # video
        ,21:0,22:0,23:0,24:0,25:0,26:0  # problem
        ,31:0,32:0,33:0,34:0            # common
        ,41:0,42:0,43:0,44:0,45:0,46:0  # click
        }
    for i in range(len(log_list)):

        series_ = log_list[i]
        
        for action_ in series_:

            action_type = str(action_)[0]
            action_type_counter[action_type] +=1
            action_counter[action_] +=1
    
    re = {'type':action_type_counter ,'action':action_counter}
    return re
@_t
def read_or_write_json(
    path:str
    ,mode:str
    ,log=None):
    ''' 读写json文件
        mode控制 read or write
    '''
    import json

    def w(__log,__path):
        if type(__log)!=type({}):
            print('ERROR : input data not a dict!')
        else:
            json.dump(__log,open(__path,'w'))
            print('SUCCESS WRITE , path ：', __path)
        return None

    def r(__log,__path)->Dict[int,list]:
        _dict = json.load(open(__path,'r'))
        return _dict

    return eval(mode)(log,path)
def dict_to_array(dict_log:dict)->list:

    ''' ->list
        函数功能:   归类后的数据被存储为dict格式 需要将其转换为list以制作数据集
                  创建空表，将每次读取到的序列追加进表内 每個序列的'-1'位置為注冊號
        note:   用list append执行很快 np.concatenate慢十倍以上'''
    i = 0
    print_key = 100000
    len_ = len(dict_log)

    dataset = []

    for k,v in dict_log.items():
        
        dataset.append(v)
        dataset[i].append(k)

        i+=1
        
        if (i%print_key)==0:
            print('already to array ',i,' areas.')
    
    print('Append finsih , dataset include ',len(dataset),' samples')

    return dataset
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
def dict_filter(_log:dict,mode:str,**kwargs)-> dict:
    ''' ->dict
        函数功能：按参数筛选字典中的数据

    '''
    def length(__log: dict,kwargs)-> dict:
        ''' ->dict
            函数功能： 按照所包含list的长度筛选字典内的对象
        '''
        if type(_log)!= type(dict(a=1)):
            return print('ERROR : input log not a dict.')


        down_ = int(kwargs['down'])
        up_   = int(kwargs['up'])
      
        useful_dict = {}
        len_list = []

        for key,value_ in _log.items():
            len_  = int(len(value_))
   
            if ((len_>= down_) & (len_<= up_)):
          
                useful_dict[key] = value_
                len_list.append(len_)
        
        import numpy as np
        print('Length filter finish , average length : ',np.mean(len_list))
        
        return useful_dict
    
    def test(__log: dict,kwargs)-> dict:
        '''->dict
            函数功能： **kwargs测试
        '''
        print('in function test!')
        print('kwargs',kwargs,'kwargs[head]',kwargs['head'])

    return eval(mode)(_log,kwargs)
def split_label(_log: list,label_rate:int)-> list:
    dataset = []
      # [
      #  ['id1','data','label'],
      #  ['id2','data','label']
      # ]

    for series in _log:
        series__ = series[:-1]
        index_ = series[-1]
        
        len_ = len(series__)
        split_point = int(0.01*(100-label_rate)*len_)

        data  = series__[:split_point]
        label = series__[split_point:]

        dataset.append([
            int(index_), data, label ])


    return dataset

def pad_series(dataset:list,pad_length = PAD_LENGTH)->list:
        
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    print(tf.__version__)

    enroll_id_list = []
    history_list = []
    future_list = []

    for sample in dataset:
        enroll_id      = int(sample[0] )
        history_s = sample[1]
        future_s  = sample[2]

        history_list.append(history_s)
        future_list.append(future_s)
        enroll_id_list.append(enroll_id)


    pad_er = keras.preprocessing.sequence.pad_sequences
    
    pad_history = pad_er(
        history_list
        ,value= int(0)
        ,padding='post' # 未知
        ,dtype=np.uint8
        ,maxlen=PAD_LENGTH   # 单条序列最大长度 由直方图观察得出
        ) 
    
    pad_future = pad_er(
        future_list
        ,value= int(0)
        ,padding='post' # 未知
        ,dtype=np.uint8
        ,maxlen=PAD_LENGTH   # 单条序列最大长度 由直方图观察得出
        ) 

    dataset = [
        enroll_id_list,
        history_list,
        future_list]
    return  dataset

_import_()
json_export_path = 'mid_export_enroll_dict.json'

enroll_dict_list_inside = read_or_write_json(
    path    = json_export_path
    ,mode   = 'r')

array = dict_to_array(enroll_dict_list_inside)
dataset = split_label(array,label_rate)

pad_dataset = pad_series(test)







