import os
from sklearn.metrics import roc_auc_score # 之前写过文档
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl # 待了解
from model import CFIN

os.environ['CUDA_VISIBLE_DEVICES'] = '2,4' # 硬件加速参数
act_feat_basic = pkl.load(open('act_feats.pkl','rb'))
user_cat_feat = ['gender', 'cluster_label']
user_num_feat = ['age','user_enroll_num']
course_cat_feat = ['course_category']
course_num_feat = ['course_enroll_num']

"""
相当于论文里特征X Context-Smoothing augmentation里的xi+gu+gc部分
@params train_feat  
@params test_feat
@return all_feat.loc[train_feat.index], all_feat.loc[test_feat.index], act_feat
"""

def feat_augment(train_feat, test_feat):
    act_feats = pkl.load(open('act_feats.pkl', 'rb'))
    # 存疑 这里把训练和测试合并 是想把一部分测试数据用于训练？？？ 看来是这样 
    # 相当于把所有行为数据用来训练了
    all_feat = pd.concat([train_feat, test_feat])

    # 计算统计值
    # gu
    # mean 基于user
    all_feat_u_mean = all_feat.groupby('username').mean()[act_feats]
    all_feat_u_mean.columns = [x+'#user#mean' for x in all_feat_u_mean.columns]
    # max 基于user
    all_feat_u_max = all_feat.groupby('username').max()[act_feats]
    all_feat_u_max.columns = [x+'#user#max' for x in all_feat_u_max.columns]
    # 追加到xi
    all_feat = pd.merge(all_feat, all_feat_u_mean, right_index=True, left_on='username')
    all_feat = pd.merge(all_feat, all_feat_u_max, right_index=True, left_on='username')
    # gc
    # 基于course
    all_feat_c_mean = all_feat.groupby('course_id').mean()[act_feats]
    all_feat_c_mean.columns = [x+'#course#mean' for x in all_feat_c_mean.columns] 
    # 基于course
    all_feat_c_max = all_feat.groupby('course_id').max()[act_feats]
    all_feat_c_max.columns = [x+'#course#max' for x in all_feat_c_max.columns]
    # 追加到xi
    all_feat = pd.merge(all_feat, all_feat_c_mean, right_index=True, left_on='course_id')
    all_feat = pd.merge(all_feat, all_feat_c_max, right_index=True, left_on='course_id')
    # 以上代码相当于论文里 Context-Smoothing 的 augmentation里的xi+gu+gc
    # ---------------------------------------- 以上无疑问----------------------------------------
    #act_feat = [] 源文件里的地雷？
    # 疑问 以下的for 是什么操作
    for f in act_feat_basic: # act_feat_basic是act_feat.pkl里的数据
        act_feat.append(f)
        act_feat.append(f+'#user#mean')
        act_feat.append(f+'#user#max')
        act_feat.append(f+'#course#mean')
        act_feat.append(f+'#course#max')

<<<<<<< HEAD
    return  all_feat.loc[train_feat.index], 
            all_feat.loc[test_feat.index], 
            act_feat # ==xi+gu+gc 从这一部分代码看是这样 但是输入的参数是act_feat.pkl里的数据还不能完全确定含义
=======
    return  all_feat.loc[train_feat.index],all_feat.loc[test_feat.index],act_feat # ==xi+gu+gc 从这一部分代码看是这样 但是输入的参数是act_feat.pkl里的数据 这还不能完全确定含义
>>>>>>> 244d211300e30e6653bec6c3531bbbc4eee456fa

"""
数据解析
@train, test, num_feat, cat_feat
"""
# dataparse在model_run里调用 给到以下参数 
# dfTrain_u,
#  dfTest_u,
#  user_num_feat ->user_cat_feat = ['gender', 'cluster_label']
#  user_cat_feat ->user_num_feat = ['age','user_enroll_num']



def dataparse(train, test, num_feat, cat_feat):
    all_data = pd.concat([train, test])
    feat_dim = 0 # 列索引的值 意思是数据从0列开始写 这傻逼非要写个dim傻逼清华
    feat_dict = dict()

    for f in cat_feat:  # 对应用户信息里的分类数据 f=性别and集群
        cat_val = all_data[f].unique() # all_data[性别].去重
        # dim指的是 存放数据的列的索引值 这傻逼等于说先写一个性别的矩阵再把集群的矩阵追加在性别的后边 
        feat_dict[f] = dict(zip(cat_val, range(feat_dim, len(cat_val)+feat_dim))) 
        feat_dim += len(cat_val) 
    
    for f in num_feat:
        feat_dict[f] = feat_dim
        feat_dim += 1

    data_indice = all_data.copy()
    data_value = all_data.copy()

    # 逐列读取all_data里的数据 匹配到分类型和数值型
    # 很恶心 没看懂
    for f in all_data.columns:
        if f in num_feat:
            data_indice[f] = feat_dict[f] # 初始的是空的
        elif f in cat_feat:
            data_indice[f] = data_indice[f].map(feat_dict[f])
            data_value[f] = 1.
        else:
            data_indice.drop(f, axis=1, inplace=True)
            data_value.drop(f, axis=1, inplace=True) 
    return feat_dim, data_indice, data_value
"""
加载数据
@直接读'feat/test_feat.csv'
@return dfTrain_u, dfTest_u, dfTrain_c, dfTest_c, dfTrain_a, dfTest_a
"""
def load_data():
    dfTrain = pd.read_csv('feat/train_feat.csv', index_col=0)
    dfTest = pd.read_csv('feat/test_feat.csv', index_col=0)
    dfTrain_u = dfTrain[user_num_feat + user_cat_feat +['truth']]
    dfTest_u = dfTest[user_num_feat + user_cat_feat +['truth']]
    
    dfTrain_c = dfTrain[course_num_feat + course_cat_feat +['truth']]
    dfTest_c = dfTest[course_num_feat + course_cat_feat +['truth']]
    dfTrain_a = dfTrain[act_feat_basic +['truth','username','course_id']]
    dfTest_a = dfTest[act_feat_basic +['truth','username','course_id']]
    
    return dfTrain_u, dfTest_u, dfTrain_c, dfTest_c, dfTrain_a, dfTest_a

"""
跑训练和测试 
@dfTrain_u, dfTest_u, dfTrain_c, dfTest_c, dfTrain_a, dfTest_a, params, act_feat
"""
def model_run( 
                 dfTrain_u, 
                 dfTest_u,
                 dfTrain_c, 
                 dfTest_c, 
                 dfTrain_a, 
                 dfTest_a, 
                 params, 
                 act_feat
                 ):
        
    u_feat_dim, u_data_indice, u_data_value = dataparse(dfTrain_u, dfTest_u, user_num_feat, user_cat_feat)
    ui_train, uv_train = np.asarray(u_data_indice.loc[dfTrain_u.index], dtype=int), np.asarray(u_data_value.loc[dfTrain_u.index], dtype=float)
    ui_test, uv_test = np.asarray(u_data_indice.loc[dfTest_u.index], dtype=int), np.asarray(u_data_value.loc[dfTest_u.index], dtype=float)
    
    params["u_feat_size"] = u_feat_dim
    params["u_field_size"] = len(ui_train[0])
        
    c_feat_dim, c_data_indice, c_data_value = dataparse(dfTrain_c, dfTest_c, course_num_feat, course_cat_feat)
    ci_train, cv_train = np.asarray(c_data_indice.loc[dfTrain_c.index], dtype=int), np.asarray(c_data_value.loc[dfTrain_c.index], dtype=float)
    ci_test, cv_test = np.asarray(c_data_indice.loc[dfTest_c.index], dtype=int), np.asarray(c_data_value.loc[dfTest_c.index], dtype=float)
    
    params["c_feat_size"] = c_feat_dim
    params["c_field_size"] = len(ci_train[0])
    
    av_train = np.asarray(dfTrain_a[act_feat], dtype=float)
    
    ai_train = np.asarray([range(len(act_feat)) for x in range(len(dfTrain_a))], dtype=int)
    av_test = np.asarray(dfTest_a[act_feat], dtype=float)
    ai_test = np.asarray([range(len(act_feat)) for x in range(len(dfTest_a))], dtype=int)

    params["a_feat_size"] = len(ai_train[0])
    params["a_field_size"] = len(ai_train[0])
    
    y_train = np.asarray(dfTrain_a['truth'], dtype=int)
    y_test = np.asarray(dfTest_a['truth'], dtype=int)
    model = CFIN(**params) # can be change
   
    # generate valid set
    train_num = len(y_train)
    indices = np.arange(train_num)
    np.random.shuffle(indices)
    split_ = int(0.8*train_num)
    ui_train_, ui_valid = ui_train[indices][:split_],ui_train[indices[split_:]]
    uv_train_, uv_valid = uv_train[indices][:split_],uv_train[indices[split_:]]
    ci_train_, ci_valid = ci_train[indices][:split_],ci_train[indices[split_:]]
    cv_train_, cv_valid = cv_train[indices][:split_],cv_train[indices[split_:]]
    ai_train_, ai_valid = ai_train[indices][:split_],ai_train[indices[split_:]]
    av_train_, av_valid = av_train[indices[:split_]],av_train[indices[split_:]]
    y_train_, y_valid = y_train[indices[:split_]], y_train[indices[split_:]] 

    # model selection
    """
    best_epoch = model.fit(ui_train_, uv_train_, ci_train_, cv_train_, ai_train_, av_train_, y_train_, ui_valid, uv_valid, ci_valid, cv_valid, ai_valid, av_valid, y_valid)
    params['epoch'] = best_epoch
    model = CFIN(**params)
    """ 
    # prediction results on test set
    
    model.fit(ui_train, uv_train, ci_train, cv_train, ai_train, av_train, y_train, ui_test, uv_test, ci_test, cv_test, ai_test, av_test, y_test, early_stopping=False)
    model.evaluate(ui_test, uv_test, ci_test, cv_test, ai_test, av_test, y_test)   

# load data
dfTrain_u, dfTest_u, dfTrain_c, dfTest_c, dfTrain_a, dfTest_a = load_data()
dfTrain_a, dfTest_a, act_feat = feat_augment(dfTrain_a, dfTest_a)

# json
params = {
    "embedding_size": 32,
    "attn_size": 16,
    "conv_size": 512,
    "context_size": 32,
    "deep_layers": [256],
    "dropout_deep": [0.9, 0.9],
    "dropout_attn": [1.],
    "activation": tf.nn.relu,
    "epoch": 27,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.00001,
    "verbose": True,
    "attn": True,
    "eval_metric": roc_auc_score,
    "random_seed": 12345
}

y_train_dfm, y_test_dfm = model_run(dfTrain_u, dfTest_u, dfTrain_c, dfTest_c, dfTrain_a, dfTest_a,params, act_feat)

