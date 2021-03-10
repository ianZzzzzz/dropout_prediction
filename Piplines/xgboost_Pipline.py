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

from xgboost import XGBClassifier # sklearn style api
from xgboost import plot_importance,plot_tree
# measure
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
RANDOM_STATE = 1

test_label = json.load(open(
        'json_file\\into_model\\list_test_label.json','r'))
test_data = json.load(open(
    'json_file\\into_model\\list_test_static_info_dataset.json','r'))
train_label = json.load(open(
    'json_file\\into_model\\list_train_label.json','r'))
train_data = json.load(open(
    'json_file\\into_model\\list_train_static_info_dataset.json','r'))

info_col = [
    'gender','birth_year' ,'edu_degree',
    'course_category','course_type','course_duration']
static_col = [
    'L_mean','L_var','L_skew','L_kurtosis',
    'S_mean','S_var','S_skew','S_kurtosis']
scene_col =[
    'video-video','video-answer','video-comment','video-courseware',
    'answer-video','answer—answer','answer-comment','answer-courseware',
    'comment-video','comment-answer','comment-comment','comment-courseware',
    'courseware-video','courseware-answer','courseware-comment','courseware-courseware']
from pandas import DataFrame
def to_df(
    sample:list,
    label: list)->DataFrame:
    
    df_data = pd.DataFrame(
        data=sample,
        columns=[
            'gender','birth_year' ,'edu_degree',
            'course_category','course_type','course_duration',
            'L_mean','L_var','L_skew','L_kurtosis',
            'S_mean','S_var','S_skew','S_kurtosis',
            'video-video','video-answer','video-comment','video-courseware',
            'answer-video','answer—answer','answer-comment','answer-courseware',
            'comment-video','comment-answer','comment-comment','comment-courseware',
            'courseware-video','courseware-answer','courseware-comment','courseware-courseware'])

    df_label = pd.DataFrame(
        data=label,
        columns=['drop_or_not']
    )
    return df_data,df_label

df_test_data ,df_test_label = to_df(
    test_data,test_label)

df_train_data ,df_train_label = to_df(
    train_data,train_label)

# 迭代次数，对于分类问题，每个类别的迭代次数，所以总的基学习器的个数 = 迭代次数*类别个数
def plot_score(name:str,x:list):
    plt.plot(x,accuracy_,color='r',label='acc')        # r表示红色
    plt.plot(x,f1_,color='g',label='f1')  #也可以用RGB值表示颜色
    plt.plot(x,recall_,color='b',label='recall')
    plt.plot(x,precision_,color='y',label='precision')
    plt.xlabel(name) 
    #plt.xlabel('max deep of tree')    #x轴表示
    plt.ylabel('%')   #y轴表示
    plt.title("result in different "+name)      #图标标题表示
    plt.legend()            #每条折线的label显示
    #######################
    #plt.savefig('test.jpg')  #保存图片，路径名为test.jpg
    plt.show()  

y_val=df_test_label

accuracy_ = []
f1_ = []
precision_ = []
recall_ = []
deeps = [0.1,0.2,0.3,0.4,0.5,3,7,10,15,20]
etas = list(range(1,10))
n_estimators = list(range(1,5))
gammas = list(range(1,10))

params = {
    'max_depth': 10,  # max_depth [缺省值=6]
    'eta': 0.05,  # learning_rate
    #  'n_estimators': n_estimator, 没差
    'silent': 1,
    'gamma':gamma/10
        # 为0打印运行信息；设置为1静默模式，不打印
}

XGB = XGBClassifier(
    silent=0 ,
    learning_rate= 0.05, # 如同学习率
    #min_child_weight=1, 
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=10, # 构建树的深度，越大越容易过拟合
    # gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    #subsample=1, # 随机采样训练样本 训练实例的子采样比
    #max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
    #colsample_bytree=1, # 生成树时进行的列采样 
    #reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #reg_alpha=0, # L1 正则项参数 
    )
model = XGB.fit(
    X=df_train_data,
    y=df_train_label)
model.evals_result()    

y_pred = model.predict(dtest)
for i in range(len(y_pred)):
    value= y_pred[i]
    
    if value >0.4:label_ = int(1)
    else:label_ = int(0)
    y_pred[i] = label_



    accuracy  = accuracy_score(y_pred,y_val)
    precision = precision_score(y_pred,y_val)
    recall    = recall_score(y_pred,y_val)
    f1        = f1_score(y_pred,y_val)

    accuracy_.append(accuracy)
    f1_.append(f1)
    precision_.append(precision)
    recall_.append(recall)

    '''
        print(
        'threshold 0.4',
        "accuracy: %.5f%%" % (accuracy*100.0),
        "precision: %.5f%%" % (precision*100.0),
        "recall: %.5f%%" % (recall*100.0),
        "f1: %.5f%%" % (f1*100.0),
        )
        
    '''

plot_score(name='gamma',x = [i/10 for i in gammas])

   
#===
params = {'verbose': False,
          'booster': 'gbtree',
          'n_estimators': 200,
          'max_depth': 4,  # max_depth [缺省值=6]
          'eta': 0.08,  # learning_rate
          'silent': 1,  # 为0打印运行信息；设置为1静默模式，不打印
        #  'nthread': 20,  # 运行时占用cpu数
          'gamma': 0.001,  # min_split_loss]（分裂最小loss）参数的值越大，算法越保守
          'min_child_weight': 1,  # 决定最小叶子节点样本权重和,缺省值=1,避免过拟合. 值过高，会导致欠拟合
          'max_delta_step': 0,
          'subsample': 0.5,  #参数控制对于每棵树，随机采样的比例 减小避免过拟合,  典型值：0.5-1，0.5代表平均采样，防止过拟合.
          'colsample_bytree': 0.8,  #树级列采样
          'colsample_bylevel': 1,  #层级列采样
          'lambda': 0.01,  # L2正则化项, 减少过拟合
          'alpha': 0.1,  # 权重的L1正则化项
          'objective': 'binary:logistic',
          'scale_pos_weight': 1,  # 通常可以将其设置为负样本的数目与正样本数目的比值
          'eval_metric': 'auc',
          'base_score': 0.5,
          }
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(params, dtrain, 42, evallist, early_stopping_rounds=10)
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

graph = xgb.to_graphviz(bst, num_trees=0, **{'size': str(10)})
graph.render(filename='xgb.dot')

fig = plt.figure(figsize=(10, 10))
ax = fig.subplots()
xgb.plot_tree(model_, num_trees=1, ax=ax)
plt.show()



y_val=df_test_label
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
plot_importance(bst,importance_type='gain')
plt.show()


