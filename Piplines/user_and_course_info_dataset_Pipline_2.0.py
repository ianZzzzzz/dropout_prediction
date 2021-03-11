import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import numpy as np
from scipy import stats
import json
TEST_OR_NOT = False
print_batch = int(1000000)
chunk_size = int(10000) # enable only when TEST_OR_NOT = True
def load(
    log_path: str,
    return_mode: str,
    encoding_='utf-8',
    read_mode = 'pd',
    columns=None,
    test=TEST_OR_NOT)-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''
   
    if read_mode == 'pd' :
        import pandas as pd
        if test ==True: # only read 10000rows 
            reader = pd.read_csv(
                log_path
                ,encoding=encoding_
                ,names=columns
                ,chunksize=chunk_size)
                
            for chunk in reader:
                # use chunk_size to choose the size of test rows instead of loop
                log = chunk
                return log.values

        else: # read full file
            log = pd.read_csv(
                log_path
                ,encoding=encoding_
                ,names=columns)
            
        print('load running!')
    if return_mode == 'df'      :return log
    if return_mode == 'ndarray' :return log.values
    if return_mode == 'list'    :return log.values.tolist()

def list_to_dict(
    list_:list,
    key_type  = 'int',
    value_type='int')-> dict:
    """[convert dict to list use the 1st cloumn make index 2nd column make value]

    Args:
        list_ (list): [shape(n,2)]
    
    Return: dict_ :w dict

    """  
    dict_ = {}

    for item_ in list_:
        index_ = int(item_[0])
        value_ = int(item_[1])
        dict_[index_] = value_
    
    return dict_

def make_dict_course_and_user_find_enroll(
    dict_enroll_find_course,
    dict_enroll_find_user,
    dict_enroll_label)->dict:
    new_dict = {}
    for e_id in dict_enroll_find_course.keys():
        
        e_id = str(e_id)
        c_id = dict_enroll_find_course[e_id]
        u_id = dict_enroll_find_user[e_id]


        new_key = str(u_id)+c_id # str + str
        label   = dict_enroll_label[int(e_id)]
        new_dict[new_key] = label
    return new_dict

dict_test_label = list_to_dict(
    load(log_path='prediction_log\\test_truth.csv',
        # columns=['e_id','drop_or_not'],
        return_mode = 'list')
        )

dict_enroll_find_user = json.load(
    open('json_file\\test_dataset\\enroll_find_user.json','r'))

dict_enroll_find_course = json.load(
    open('json_file\\test_dataset\\enroll_find_course.json','r'))




list_test_data = json.load(open(
    'json_file\\into_model\\list_test_static_info_dataset.json','r'))
list_test_label = json.load(open(
    'json_file\\into_model\\list_test_label.json','r'))
list_e_id_in_testDataset_ordered = list(dict_test_label.keys())

dict_user_find_course = {}
dict_course_find_user = {}

for e_id in list_e_id_in_testDataset_ordered:
    e_id = str(e_id)
    u_id = dict_enroll_find_user[e_id]
    c_id = dict_enroll_find_course[e_id]

    try:
        dict_course_find_user[c_id].append(u_id)
    except:
        dict_course_find_user[c_id] = []
        dict_course_find_user[c_id].append(u_id)
        
    try:
        dict_user_find_course[u_id].append(c_id)
    except:
        dict_user_find_course[u_id] = []
        dict_user_find_course[u_id].append(c_id)



dict_u_and_c_find_label = make_dict_course_and_user_find_enroll(
    dict_enroll_label = dict_test_label,
    dict_enroll_find_course=dict_enroll_find_course,
    dict_enroll_find_user =dict_enroll_find_user )

dict_user_dropout_rate = {}
for u_id,courses in dict_user_find_course.items():
    label_list = []
    for c_id in courses:
        key = str(u_id)+str(c_id)
        label = dict_u_and_c_find_label[key]
        label_list.append(label)
    dropout_rate = int((
        sum(label_list)/len(label_list)
        )*100)
    
    course_amont = len(courses)
    dict_user_dropout_rate[u_id] = [
        course_amont,
        dropout_rate ]
    
 #   json.dump(dict_user_dropout_rate,open('json_file\\info_2.0\\dict_user_dropout_rate.json','w'))


dict_course_dropout_rate = {}
for c_id,users in dict_course_find_user.items():
    label_list = []
    for u_id in users:
        key = str(u_id)+str(c_id)
        label = dict_u_and_c_find_label[key]
        label_list.append(label)
    
    user_amount = len(users)
    dropout_rate = int((
        sum(label_list)/len(label_list)
        )*100)
    dict_course_dropout_rate[c_id] = [
        user_amount,
        dropout_rate ]

    
  #  json.dump(dict_course_dropout_rate,open('json_file\\info_2.0\\dict_course_dropout_rate.json','w'))


# asemble data
row = 0
test_data = list_test_data
for e_id in list_e_id_in_testDataset_ordered:
    e_id = str(e_id)
    u_id = dict_enroll_find_user[e_id]
    c_id = dict_enroll_find_course[e_id]

    c_amount = dict_course_dropout_rate[c_id][0]
    u_amount = dict_user_dropout_rate[u_id][0]
    c_d_rate = dict_course_dropout_rate[c_id][1]
    u_d_rate = dict_user_dropout_rate[u_id][1]

    test_data[row].append(c_amount)
    test_data[row].append(c_d_rate)
    test_data[row].append(u_amount)
    test_data[row].append(u_d_rate)

    row+=1


data = [
    # student_amount
    # course_amount
    # dropout rate of course
    # dropout rate of user
    # cluster
    ]




   

if len(data)==len(list_test_data):
    print('data len correct!')
else:
    print('data length unmatch!!!!!')

for row in range(len(list_test_data)):
    #ori_data = list_test_data[row]
    new_info = data[row]

    # 不用extend 怕指针出错
    for item in new_info:
        list_test_data[row].append(item)

json.dump(
    list_test_data,
    open('list_test_static_info_dataset_2.0.json','w'))



