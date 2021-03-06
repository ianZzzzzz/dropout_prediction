import time
import numpy as np
import json
import pandas as pd

import os
from sklearn.utils import shuffle # sklean 中有数据集打乱方法，为了增加机器学习的鲁棒性
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# preprocrss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# model
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance,plot_tree
# measure
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

test_label = json.load(open(
        'json_file\\into_model\\list_test_label.json','r'))
test_data = json.load(open(
    'json_file\\into_model\\list_test_static_info_dataset.json','r'))



train_label = json.load(open(
    'json_file\\into_model\\list_train_label.json','r'))
train_data = json.load(open(
    'json_file\\into_model\\list_train_static_info_dataset.json','r'))


X = np.array(train_data)
y = np.array(train_label)

test_data_ = list_test_static_info_dataset
X = np.array(test_data_)
y = np.array(test_label)


x_val = np.array(test_data)
y_val = np.array(test_label)

X_train, X_test, y_train, y_test_np = train_test_split(
      X
    , y
    , test_size=0.15
    , random_state=123
    , stratify=y)



dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test,y_test_np)
dval = xgb.DMatrix(x_val,y_val)
# 迭代次数，对于分类问题，每个类别的迭代次数，所以总的基学习器的个数 = 迭代次数*类别个数
num_rounds = 40
params = {}
watchlist = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params, dtrain, num_rounds,watchlist) # xgboost模型训练


#===
params = {'verbose': False,
          'booster': 'gbtree',
          'n_estimators': 200,
          'max_depth': 9,  # max_depth [缺省值=6]
          'eta': 0.08,  # learning_rate
          'silent': 1,  # 为0打印运行信息；设置为1静默模式，不打印
          'nthread': 20,  # 运行时占用cpu数
          'gamma': 0.0,  # min_split_loss]（分裂最小loss）参数的值越大，算法越保守
          'min_child_weight': 5,  # 决定最小叶子节点样本权重和,缺省值=1,避免过拟合. 值过高，会导致欠拟合
          'max_delta_step': 0,
          'subsample': 1,  #参数控制对于每棵树，随机采样的比例 减小避免过拟合,  典型值：0.5-1，0.5代表平均采样，防止过拟合.
          'colsample_bytree': 0.8,  #树级列采样
          'colsample_bylevel': 1,  #层级列采样
          'lambda': 0.1,  # L2正则化项, 减少过拟合
          'alpha': 0.1,  # 权重的L1正则化项
          'objective': 'binary:logistic',
          'scale_pos_weight': 1,  # 通常可以将其设置为负样本的数目与正样本数目的比值
          'eval_metric': 'auc',
          'base_score': 0.5,
          }
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, 100, evallist, early_stopping_rounds=10)
bst.save_model('xgb.model')

from xgboost import plot_tree
import matplotlib.pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# load model.
bst = xgb.Booster()
bst.load_model('xgb.model')
# plot
plot_tree(bst,fmap='', num_trees=0, rankdir='UT', ax=None)
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
xgb.plot_tree(model_, num_trees=1, ax=ax)
plt.show()



y_val=y_test_np
for v in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    y_pred = model.predict(dtest)
    for i in range(len(y_pred)):
        value= y_pred[i]
        
        if value >v:label_ = int(1)
        else:label_ = int(0)
        y_pred[i] = label_



    accuracy  = accuracy_score(y_pred,y_val)
    precision = precision_score(y_pred,y_val)
    recall    = recall_score(y_pred,y_val)
    f1        = f1_score(y_pred,y_val)


    print(
        'threshold',v,
        "accuracy: %.5f%%" % (accuracy*100.0),
        "precision: %.5f%%" % (precision*100.0),
        "recall: %.5f%%" % (recall*100.0),
        "f1: %.5f%%" % (f1*100.0),
        )



#---------------------------------

X = pd.DataFrame(train_data)
y = pd.DataFrame(train_label)


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

#标准化
std=StandardScaler()
mm = MinMaxScaler()
pipeline=make_pipeline(std,mm)
X = pipeline.fit_transform(X)#训练模型

X_new = SelectKBest(chi2, k=10).fit_transform(X, y)



X_train, X_test, y_train, y_test_np = train_test_split(
      X
    , y
    , test_size=0.15
    , random_state=123
    , stratify=y)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test,y_test_np)
# 迭代次数，对于分类问题，每个类别的迭代次数，所以总的基学习器的个数 = 迭代次数*类别个数
num_rounds = 40
params = {}
'''params = {
    'booster':'gbtree'
    ,'max_depth':6
    , 'eta':0.03
    , 'silent':1 # activate print
    , 'objective':'binary:logistic' }
'''
watchlist = [(dtrain,'train'),(dtest,'test')]
model = xgb.train(params, dtrain, num_rounds,watchlist) # xgboost模型训练

y_test = []
for i in range(len(y_test_np)):
    label_=int( y_test_np.values[i,0])
    y_test.append(label_)
np_y_pred = model.predict(dtest)
y_pred = []
for i in range(len(np_y_pred)):
    value= np_y_pred[i]
    label_ = None
    if value >0.5:label_ = int(1)
    else:label_ = int(0)
    y_pred.append(label_)


accuracy = accuracy_score(y_pred,y_test)

accuracy = precision_score(y_pred,y_test)
print('threshold','0.1',"precision_score: %.5f%%" % (accuracy*100.0))

#  ,, static
'''# info: 0  gender
            1,birth_year
            2,edu_degree
            3,course_category
            4,course_type
            5,course_duration
    long static       
            6 mean
            7 var
            8 偏度
            9 峰度 
    short static       
            10 mean
            11 var
            12 偏度
            13 峰度 
            '''
plot_importance(model)
plt.show()



for v in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    np_y_pred = model.predict(dtest)
    y_pred = []
    for i in range(len(np_y_pred)):
        value= np_y_pred[i]
        label_ = None
        if value >v:label_ = int(1)
        else:label_ = int(0)
        y_pred.append(label_)


    accuracy = f1_score(y_test,y_pred)
    print('threshold',v,"f1_score: %.5f%%" % (accuracy*100.0))

dtest = xgb.DMatrix(test_data,test_label)
for v in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    np_y_pred = model.predict(dtest)
    y_pred = []
    for i in range(len(np_y_pred)):
        value= np_y_pred[i]
        
        if value >v:label_ = int(1)
        else:label_ = int(0)
        y_pred.append(label_)


    accuracy = precision_score(y_test,y_pred)
    print('threshold',v,"precision_score: %.5f%%" % (accuracy*100.0))
