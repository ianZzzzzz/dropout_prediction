'''自监督文本分类
'''
import os

import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing   import FunctionTransformer
from sklearn.linear_model    import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics         import f1_score
def load_split_data():
    data = fetch_20newsgroups(subset='train', categories=None)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))
    print()

    Sample, Label = data.data, data.target
    Sample_train, Sample_test, Label_train, Label_test = train_test_split(Sample, Label)

    dataset = dict(
        train_S= Sample_train
        ,train_L= Label_train
        ,test_S= Sample_test
        ,test_L= Label_test)

    return dataset

def set_parameters():
    sdg_params = dict(alpha=1e-5, penalty='l2', loss='log')
    vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
    return sdg_params,vectorizer_params

def set_pipline():
        # Supervised Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(**sdg_params)),   ])
    # SelfTraining Pipeline
    st_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),])
    # LabelSpreading Pipeline
    ls_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        # LabelSpreading does not support dense matrices
        ('todense', FunctionTransformer(lambda x: x.todense())),
        ('clf', LabelSpreading()),])
    
    piplines = dict(ori = pipeline ,self = st_pipeline,ls = ls_pipeline)
    return piplines


def eval_and_print_metrics(
    clf
    , Sample_train, Label_train
    , Sample_test, Label_test
    ):
    print("Number of training samples:", len(Sample_train))
    print("Unlabeled samples in training set:",
          sum(1 for x in Label_train if x == -1))
    clf.fit(Sample_train, Label_train)
    print('fit')
    y_pred = clf.predict(Sample_test)
    print('predict')
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(Label_test, y_pred, average='micro'))
    print("-" * 10)
    print()




# 加载并切片数据
dataset = load_split_data()
# 设置模型参数
sdg_params , vectorizer_params    = set_parameters()
# 设置模型流水线
self_sup_pipeline = set_pipline()['self']
# 运行模型打印结果
dataset['train_L'] = np.array(list(map(lambda x : -1,dataset['train_L'])))
dataset['train_L']
print("SelfTrainingClassifier on 20% of the training data (rest "
        "is unlabeled):")
eval_and_print_metrics(
    self_sup_pipeline
    , dataset['train_S']
    , dataset['train_L']
    , dataset['test_S']
    , dataset['test_L'])



print("Supervised SGDClassifier on 100%% of the data:")

# select a mask of 20% of the train dataset
y_mask = np.random.rand(len(Label_train)) < 0.2


# X_20 and y_20 are the subset of the train dataset indicated by the mask
X_20, y_20 = map(list, zip(*((x, y)
                    for x, y, m in zip(Sample_train, Label_train, y_mask) if m)))
print("Supervised SGDClassifier on 20% of the training data:")
eval_and_print_metrics(pipeline, X_20, y_20, Sample_test, Label_test)

# set the non-masked subset to be unlabeled

if 'CI' not in os.environ:
    # LabelSpreading takes too long to run in the online documentation
    print("LabelSpreading on 20% of the data (rest is unlabeled):")
    eval_and_print_metrics(ls_pipeline, Sample_train, Label_train, Sample_test, Label_test)



def dict_filter(_log:dict,mode:str,**kwargs)-> dict:
    
    def length(__log: dict,kwargs)-> dict:
        
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
        print('in function test!')
        print('kwargs',kwargs,'kwargs[head]',kwargs['head'])

    return eval(mode)(_log,kwargs)

def dict_to_array(dict_log:dict)->list:

    ''' 函数功能:   归类后的数据被存储为dict格式 需要将其转换为list以制作数据集
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
            int(index_),
            data,
            label
            ])

    return dataset


from typing import List, Dict
def read_or_write_json(
    path:str
    ,mode:str
    ,log: Dict[int,list]):
    ''' 读写json文件
        mode控制 read or write
    '''
    import json
    def w(__log,__path):
        print('w')
        if type(__log)!=type(dict):
            print('ERROR : input data not a dict!')
        else:
            print('dump')
            json.dump(__log,open(__path,'w'))
            print('SUCCESS WRITE , path ：', __path)
        return None

    def r(__log,__path)->Dict[int,list]:
        _dict = json.load(open(__path,'r'))
        return _dict

    return eval(mode)(log,path)


def w(__log,__path):
    
    print('w')
    if type(__log)!=type({}):
        print('ERROR : input data not a dict!')
    else:
        print('dump')
        json.dump(__log,open(__path,'w'))
        print('SUCCESS WRITE , path ：', __path)
    return None

__log = t
__path = 'a_test.json'

t = {'776':[1,221,212,2]}

w(__log = t,__path = 'a_test.json')