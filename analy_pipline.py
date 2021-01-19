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
def to_int(Sample:list)->list:
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

    #num_clusters = 2 #聚为四类，可根据需要修改

    km = KMeans(n_clusters=num_clusters)

    km.fit(Sample_Vector)

    Sample_labels = km.labels_.tolist()
    return Sample_labels

json_export_path = 'washed_log_list.json'

reader = read_or_write_json(
    path= json_export_path
    ,mode = 'r')
log_list = reader
useful_list = cut_toolong_tooshort(log_list,up = 2000,down = 100)
# use n-gram can use more useful data
plot_histogram(useful_list) 
avg_series_len = find_avg_length_of_series(useful_list)
Sample = to_int(useful_list)
Sample_Vector = count_words(Sample)
Sample_Label = k_mean(Sample_Vector,5)

def ngram_vectorize(train_texts, train_labels, val_texts):
    
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    # Vectorization parameters
    # Range (inclusive) of n-gram sizes for tokenizing text.
    NGRAM_RANGE = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 20000

    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = 'word'

    # Minimum document/corpus frequency below which a token will be discarded.
    MIN_DOCUMENT_FREQUENCY = 2
    
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val

Sample_Label = np.array(Sample_Label)
for i in set(Sample_Label):
    mask = Sample_Label==i
    print('class ',i,' : ',len(Sample_Label[mask]))

