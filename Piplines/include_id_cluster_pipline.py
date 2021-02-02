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
 
def dict_to_list(dict_log:dict,drop_key = False)->list:

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
            v.append(int(k))
            dataset.append(v)
            
            i+=1
    else:
        print(' Series data only.')
        
        for k,v in dict_log.items():
            dataset.append(v)
            i+=1

    print('Append finsih , dataset include ',len(dataset),' samples')

    return dataset


def list_filter(_log:list,mode:str,**kwargs)-> list:
    """按参数筛选字典中的数据

    Args:
        _log (list): [log]
        mode (str): [筛选模式]

    Returns:
        list: [log]
    """    
    def find_avg_length_of_series(log_list: list)->int:
        len_array = np.zeros( len(log_list)+1,dtype = np.uint32)
        for i in range(len(log_list)):
            length =  len(log_list[i])
            len_array[i] = length

        mean_length =int( np.mean(len_array))
        return mean_length
    
    def length(__log: dict,kwargs)-> dict:
        ''' ->list
            函数功能： 按照所包含list的长度筛选list内的list
        '''
        if type(_log)!= type(list([])):
            return print('ERROR : input log not a list.')


        down_ = int(kwargs['down'])
        up_   = int(kwargs['up'])
      
        useful_list = []
        len_list = []

        for _series in _log:
            len_  = int(len(_series))
   
            if ((len_>= down_) & (len_<= up_)):
          
                useful_list.append(_series)
                len_list.append(len_)
        
        import numpy as np
        print('Length filter finish , average length : ',int(np.mean(len_list)))
        
        return useful_list
    
    def to_str(_log:list,kwargs)->list:
        Sample = _log
        Sample = list(map(lambda x: str(x), Sample))
        return Sample

    def test(__log: list,kwargs)-> list:
        '''->dict
            函数功能： **kwargs测试
        '''
        print('in function test!')
        print('kwargs',kwargs,'kwargs[head]',kwargs['head'])

    return eval(mode)(_log,kwargs)

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
list_enroll_id_in_tail = dict_to_list(enroll_dict_list_inside,drop_key= False)



# wash


useful_list = list_filter(mode = 'length',
    _log= list_enroll_id_in_tail,
    up = 1000,
    down = 100)

Sample = list_filter(mode = 'to_str',
    _log= useful_list)

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

