import time
import numpy as np
import json
import pandas as pd

import os
from sklearn.utils import shuffle # sklean 中有数据集打乱方法，为了增加机器学习的鲁棒性
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance,plot_tree
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



X_train, X_test, y_train, df_y_test = train_test_split(
      X
    , y
    , test_size=0.15
    , random_state=123
    , stratify=y)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test,df_y_test)
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
for i in range(len(df_y_test)):
    label_=int( df_y_test.values[i,0])
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
