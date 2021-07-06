"""
@ 'train_features.csv' 'test_features.csv'
@ 提取  Z（u,c）
    提取用户年龄、性别、教育水平、集群、
    提取课程类别
@ 'train_feat.csv' 'test_feat.csv'
"""
import pandas as pd
import numpy as np
import pickle as pkl
import math
from sklearn.preprocessing import StandardScaler

train_feat= pd.read_csv('train_features.csv', index_col=0)
test_feat= pd.read_csv('test_features.csv', index_col=0)
all_feat = pd.concat([train_feat, test_feat])
# 用户信息
user_profile = pd.read_csv('user_info.csv', index_col='user_id')
# extract user age
birth_year = user_profile['birth'].to_dict()
"""
@ birth_year
@ 对于用户信息里记录的小于十岁或大于七十岁的用户 统一归为0岁 
    改进时可以做一些调整
@ return age
"""
def age_convert(y):
    if y == None or math.isnan(y):
        return 0
    a = 2018 - int(y)
    if a> 70 or a< 10:
        a = 0
    return a
# 此处get取值 
all_feat['age'] = [age_convert( birth_year.get( int(u) ,None) ) for u in all_feat['username']]
# extract user gender
user_gender = user_profile['gender'].to_dict()
"""
@ user_gender
@ 女 return 1 
@ 男 return 2
@ 未知 return 0
"""
def gender_convert(g):
    if g == 'm':
        return 1
    elif g == 'f':
        return 2
    else:
        return 0
#
all_feat['gender'] = [gender_convert(user_gender.get(int(u),None)) for u in all_feat['username']]

user_edu = user_profile['education'].to_dict()
#存疑 index这里怎么用 已改动
def edu_convert(x):
    edus = ["Bachelor's","High", "Master's", "Primary", "Middle","Associate","Doctorate"]
    #if x == None or or math.isnan(x):
    #    return 0
    if not isinstance(x, str):
        return 0
    ii = edus.index(x)
    return ii+1


# for u in all_feat['username']:
#     index = edu_convert(user_edu.get(int(u), None))
#     all_feat['education'].push(index)

all_feat['education'] = [edu_convert(user_edu.get(int(u), None)) for u in all_feat['username']]# 为了占位符写的
#-----------------------------------------------------------------------------------------------------------
# 计算每个用户名下有多少门课程
user_enroll_num = all_feat.groupby('username').count()[['course_id']]
# 计算每门课程下有多少名用户
course_enroll_num = all_feat.groupby('course_id').count()[['username']]
# 写列名
user_enroll_num.columns = ['user_enroll_num']
course_enroll_num.columns = ['course_enroll_num']
# 合并到all_feat
all_feat = pd.merge(all_feat, user_enroll_num, left_on = 'username', right_index = True)
all_feat = pd.merge(all_feat, course_enroll_num, left_on='course_id', right_index=True)
#--------------------------------------------------------------------------------------------
#待解释

#extract user cluster 用户集群 
user_cluster_id = pkl.load(open('cluster/user_dict','r'))
cluster_label = np.load('cluster/label_5_10time.npy')
all_feat['cluster_label'] = [cluster_label[user_cluster_id[u]] for u in all_feat['username']]

#extract course category  读取 course_info.csv 以id列为索引
course_info = pd.read_csv('course_info.csv', index_col='id')
# 把课程分成以下几类
en_categorys = [ 
      'math',
      'physics',
      'electrical', 
      'computer',
      'foreign language',
      'business',
      'economics',
      'biology',
      'medicine',
      'literature',
      'philosophy',
      'history',
      'social science', 
      'art',
      'engineering',
      'education',
      'environment',
      'chemistry'
      ]
#存疑
def category_convert(cc):  
    # isinstance 判断cc的类型 是str返回True 反之返回Flase
    if isinstance(cc, str):
        for i, c in zip(range(len(en_categorys)), en_categorys): # en_categorys 是课程的分类名
            if cc == c:
                return i+1
    else:
        return 0
    # reformat
    # if isinstance(cc, str) is false:
    #     return 0
    # for key,value in encategorys:
    #     if value == cc
    #         return i+1
    # return 0


category_dict = course_info['category'].to_dict()

# 对dict进行get就是取值， 第二个参数是默认返回值 none 意思就是当值不存在时返回none
all_feat['course_category'] = [
    category_convert( category_dict.get(str(x), None) ) for x in all_feat['course_id']
    ]
# act_feat从最原始的训练数据 train
act_feats = [c for c in train_feat.columns if 'count' in c or 'time' in c or 'num' in c]
# 以二进制写入到pkl文件
pkl.dump(act_feats, open('act_feats.pkl','wb'))

num_feats = act_feats + ['age','course_enroll_num','user_enroll_num']
# 将数据标准化
scaler= StandardScaler()
newX = scaler.fit_transform(all_feat[num_feats])
print(newX.shape)

for i, n_f in enumerate(num_feats):
    all_feat[n_f] = newX[:,i]   



#写入csv文件
all_feat.loc[train_feat.index].to_csv('train_feat.csv')
all_feat.loc[test_feat.index].to_csv('test_feat.csv')

