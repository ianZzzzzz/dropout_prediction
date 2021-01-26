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

def read_or_write_json(
    path:str
    ,mode:str
    ,log=None):
    ''' 读写json文件
        
        mode== 'r' -> read  return dict (enroll_id : log_list)

        mode=='w' -> write return None
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
def dict_to_array(dict_log:dict,drop_key = False)->list:

    ''' 
        函数功能:
            将dict格式数据集转换为list格式数据集
        
        return：list ( [ 
                    [log_list_1,enroll_id_1],
                    [log_list_2,enroll_id_2],
                    [log_list_n,enroll_id_n] ])

                -1位置为dict数据集的key —>enroll_id 

        note:   用list append执行很快 np.concatenate慢十倍以上'''
    i = 0
    print_key = 100000
    len_ = len(dict_log)

    dataset = []
    if drop_key == False:
        print('Inclouding enroll id in [-1] position.')

        for k,v in dict_log.items():
            
            dataset.append(v)
            
            i+=1
    else:
        print(' Series data only.')
        
        for k,v in dict_log.items():
            v = v[:-1]
            dataset.append(v)

            i+=1

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
def to_str(Sample:list)->list:
    
    Sample = list(map(lambda x: str(x), Sample))
    return Sample

@_t
def count_words(Sample:list)-> list:
    '''
        函数功能：根据传入的参数选择相应的方式向量化文本 
            1 word bag
            2 n-gram
            
    '''
    # 转换成字符串 因为sklrarn的库只接受str类型的样本
    
    from sklearn.feature_extraction.text import CountVectorizer
    # 实例化
    vectorizer = CountVectorizer()
    # 遍历所有样本来建立词表
    bag = vectorizer.fit(Sample)
    # 向量化
    vector = vectorizer.transform(Sample)

    return vector.toarray().tolist()

def tf_idf_width(Sample:list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 实例化
    vectorizer = TfidfVectorizer()
    # 标记并建立索引
    vectorizer.fit(Sample)
    # 编码文档
    vector = vectorizer.transform(Sample)

def k_mean(Sample: list,num_clusters):
    from sklearn.cluster import KMeans

   
    km = KMeans(n_clusters=num_clusters)

    km.fit(Sample_Vector)

    Sample_labels = km.labels_.tolist()
    return Sample_labels

# load
json_export_path = 'Piplines\\mid_export_enroll_dict.json'
enroll_dict_list_inside = read_or_write_json(
    path    = json_export_path
    ,mode   = 'r')
non_id_array = dict_to_array(enroll_dict_list_inside,drop_key= True)

# wash
useful_list = cut_toolong_tooshort(non_id_array,up = 5000,down = 100)
Sample = to_str(useful_list)
Sample_Vector = count_words(Sample)

# cluster
Sample_Label = k_mean(Sample_Vector,4)
cluster_3 = []
cluster_2 = []
cluster_1 = []
cluster_0 = []

Sample = useful_list
for i in range(len(Sample)):
    
    if Sample_Label[i] ==0:
        cluster_0.append(Sample[i])
    if Sample_Label[i] ==1:
        cluster_1.append(Sample[i])
    if Sample_Label[i] ==2:
        cluster_2.append(Sample[i])
    if Sample_Label[i] ==3:
        cluster_3.append(Sample[i])

final = [
    len(cluster_0),len(cluster_1)
    ,len(cluster_2),len(cluster_3)]
# final->[15473, 102, 293, 1722]
# choose cluster_0

final_data = [
    cluster_0,cluster_1
    ,cluster_2,cluster_3]

i = 0
for c in final_data:
    print('cluster ',i,' : ',find_avg_length_of_series(c))
    i+=1
''' cluster  0  :  1232.7522295463357      
    cluster  1  :  14680.85436893204       
    cluster  2  :  6430.272108843537       
    cluster  3  :  4971.98142774231 
'''
# generate use clustelenr_0

