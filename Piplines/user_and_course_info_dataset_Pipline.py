import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import numpy as np
import json
TEST_OR_NOT = False
print_batch = int(1000000)
chunk_size = int(10000) # enable only when TEST_OR_NOT = True
def load(
    log_path: str,
    read_mode: str,
    return_mode: str,
    encoding_='utf-8',
    columns=None,
    test=TEST_OR_NOT)-> ndarray or DataFrame:
    '''读取csv文件 返回numpy数组'''
    #if read_mode == 'cudf':import cudf as pd
    if read_mode == 'pandas' :
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
    if return_mode == 'df':return log
    if return_mode == 'values':return log.values


##========== make user and course info dataset =============#

def user_info_list_to_dict(list_:list)-> dict:
    """[convert list to dict  use the 1st cloumn make index 2nd column make value]

    Args:
        list_ (list): [shape(n,2)]
    
    Return:dict_[u_id] = [gender,edu,birth]

    """  
    dict_ = {}
    gender_replace_dict ={
         'nan' :0
        , 'male' :1
        , 'female' :2}    
    education_degree_replace_dict = {
          'nan' : 0
        , 'Primary' :1
        , 'Middle' :2
        , "Bachelor's" :3
        , "Master's" :4
        , 'Associate' :5
        , 'High' :6
        , 'Doctorate' :7 
        , 'education' :8
        }

    for item_ in list_:
        u_id = int(item_[0])
        gender = item_[1]
        edu    = item_[2]
        birth  = item_[3]

        try:
            gender = gender_replace_dict[gender]
        except:
            gender = int(0)
        
        try:
            edu = education_degree_replace_dict[edu]
        except:
            edu = int(0)
        try:
            birth = int(birth)
        except:
            birth = int(0)
        
        dict_[u_id] = [gender,edu,birth]
        
    
    return dict_

def course_info_list_to_dict(list_:list)-> dict:
    """[convert list to dict  use the 1st cloumn make index 2nd column make value]

    Args:
        list_ (list): [shape(n,2)]
    
    Return: dict_[c_id] = [
            course_category
            ,course_type
            ,start_time
            ,end_time]
        

    """  
    dict_ = {}
    cate_replace_dict = {
        'nan':0
        , 'social science':1
        , 'business':2
        , 'electrical':3
        , 'chemistry':4
        , 'math':5
        , 'environment':6
        , 'biology':7
        , 'history':8
        , 'education':9
        , 'medicine':10
        , 'economics':11
        , 'art':12
        , 'physics':13
        , 'foreign language':14
        , 'literature':15
        , 'philosophy':16
        , 'engineering':17
        , 'computer':18
        }
    
    for item_ in list_:
        c_id = item_[1] # str
        start_time = item_[2]
        end_time    = item_[3]
        course_type  = int(item_[4])
        course_category = item_[5]

        try:
            course_category = cate_replace_dict[course_category]
        except:
            course_category = int(0)
        
        '''if start_time is np.nan:
            start_time = 0
        else:
            start_time = np.datetime64(start_time)
        if end_time is np.nan:
            end_time = 0
        else:
            end_time = np.datetime64(end_time)
        '''
       
        dict_[c_id] = [
            course_category
            ,course_type
            ,start_time
            ,end_time]
        
    
    return dict_

def make_info_dataset(return_mode: str
    ,enroll_find_course
    ,enroll_find_user
    ,dict_c_info
    ,dict_u_info)->dict or list:
    """[summary]

    Args:
        return_mode (str): [description]
        enroll_find_course ([type]): [description]
        enroll_find_user ([type]): [description]
        dict_c_info ([type]): [description]
        dict_u_info ([type]): [description]

    Returns:
        dict or list: [info_ = [
            gender
            ,birth_year
            ,edu_degree
            ,course_category
            ,course_type
            ,course_duration
        ]]
    """    

    dict_enroll_info = {}
    list_enroll_info = []
    for e_id in enroll_find_course.keys():
        
        c_id = enroll_find_course[e_id]
        u_id = str(enroll_find_user[e_id])
        
        # user info
        gender     = dict_u_info[u_id][0]
        edu_degree  = dict_u_info[u_id][1]
        birth_year = dict_u_info[u_id][2]

        # course info
        course_category   = dict_c_info[c_id][0]
        course_type = dict_c_info[c_id][1]

        course_start = dict_c_info[c_id][2]
        course_end = dict_c_info[c_id][3]

        

        if (course_start is np.nan) or (course_end is np.nan):
            course_duration = 0
        else:
            start_ = np.datetime64(course_start)
            end_   = np.datetime64(course_end)

            course_duration =  int(
                            ( end_ - start_ ).item().total_seconds() )
                        
        info_ = [
            gender
            ,birth_year
            ,edu_degree
            ,course_category
            ,course_type
            ,course_duration
        ]
        if return_mode == 'dict':
            dict_enroll_info[int(e_id)] = info_
        else: 
            if  return_mode == 'list':
                list_enroll_info.append(info_)
        
    if return_mode == 'dict':
        import json
        json.dump(dict_enroll_info,open('json_file\\dict_enroll_info_full.json','w'))
        return    dict_enroll_info
    else: 
        if  return_mode == 'list':
            return    list_enroll_info
   
c_info_path = 'course_info.csv'
c_info_col = ['id','course_id','start_time','end_time','course_type','course_category']
C_INFO_NP = load(
    log_path =c_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =c_info_col
    )

u_info_path = 'user_info.csv'
u_info_col = ['user_id','gender','education_degree','birth_year']
U_INFO_NP = load(
    log_path =u_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =u_info_col)

dict_u_info = user_info_list_to_dict(U_INFO_NP[1:,:])
# json.dump(dict_u_info,open('dict_u_info.json','w'))
dict_c_info = course_info_list_to_dict(C_INFO_NP[1:,:])
# json.dump(dict_c_info,open('dict_course_info.json','w'))
dict_c_info = json.load(
    open('dict_course_info.json','r'))
dict_u_info = json.load(
    open('dict_u_info.json','r'))
enroll_find_course_test = json.load(
    open('json_file\\test_dataset\\enroll_find_course.json','r'))
enroll_find_user_test = json.load(
    open('json_file\\test_dataset\\enroll_find_user.json','r'))
enroll_find_course_train = json.load(
    open('json_file\\train_dataset\\enroll_find_course.json','r'))
enroll_find_user_train = json.load(
    open('json_file\\train_dataset\\enroll_find_user.json','r'))

dict_enroll_info_full = make_info_dataset(
    return_mode= 'dict'
    ,dict_c_info=dict_c_info
    ,dict_u_info= dict_u_info
    ,enroll_find_course= {
        **enroll_find_course_test,
        **enroll_find_course_train}
    ,enroll_find_user= {
        **enroll_find_user_test,
        **enroll_find_user_train
    }
    )
#================ info dataset finish ==================#
dict_enroll_info_full = json.load(open('json_file\\dict_enroll_info_full.json','r'))
#================ assemble train and test dataset ===============#

def caculate_time_interval(dict_log)-> dict:


    def static_(interval_list):
            
            if len(interval_list) > int(3):
                R = interval_list
                R_mean = np.mean(R) # 计算均值
                R_var = np.var(R)   # 计算方差
                R_sc = np.mean((R - R_mean) ** 3)  #计算偏斜度
                R_ku = np.mean((R - R_mean) ** 4) / pow(R_var, 2) #计算峰度

                R_sc = np.abs(R_sc)
                R_ku = np.abs(R_ku)

                static_list = [
                    round(R_mean,2)
                    ,round(R_var,2)
                    ,round(R_sc,2)
                    ,round(R_ku,2)]

            else:
                static_list = [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                ]
                
            
            return static_list

    time_interval_dict   = {}
    static_interval_dict = {}
    enroll_scene_dict    = {}
    i = 0
    for e_id,list_log in dict_log.items():
        # 1 time interval
        long_interval_list  = []
        short_interval_list = []
        # 2 state transition
        scene_dict = {}
        for row in range(len(list_log)-1):
            
            row_next = list_log[row+1]
            row_now  = list_log[row]

            now_time = row_now[0]
            now_action = row_now[1]

            now_object = row_now[2]
            now_session = row_now[3]
            next_time = row_next[0]
            next_action = row_next[1]
            next_object = row_next[2]
            next_session = row_next[3]
            try:
                time_interval = int(next_time - now_time)
                if time_interval >int(60*30): # half hour 
                    long_interval_list.append(time_interval)
                else:
                    short_interval_list.append(time_interval)
            except:
                print(next_time,now_time)
        
        short_static = static_(short_interval_list)
        long_static  = static_(long_interval_list)
        i+=1
        
        if (i%1000)==0:print(i)

        # time_interval_dict[int(e_id)] = interval_list
        long_static.extend(short_static)
        static_interval_dict[int(e_id)] = long_static
        #enroll_scene_dict[int(e_id)] = scene_dict
        # break
    return static_interval_dict
def label_list_to_dict(list_:list)-> dict:
    """[convert list to dict  use the 1st cloumn make index 2nd column make value]

    Args:
        list_ (list): [shape(n,2)]
    
    Return:dict_[e_id] = label

    """  
    dict_ = {}
    
    for item_ in list_:
        e_id = int(item_[0])
        label_ = int(item_[1])
        
        dict_[e_id] = label_
    
    return dict_

list_train_label = pd.read_csv('prediction_log\\train_truth.csv').values.tolist()
list_test_label = pd.read_csv('prediction_log\\test_truth.csv').values.tolist()

dict_train_label = label_list_to_dict(list_train_label)
dict_test_label = label_list_to_dict(list_test_label)

dict_log_test = json.load(
    open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_log_test_sec_version.json'
        ,'r')
        ) 
dict_static_intervel_test = caculate_time_interval(dict_log_test)
  
i = 0
list_test_static_info_dataset = []
# c_inf0,u_info,long_static,short_static
list_test_label = []
for e_id,static_ in dict_static_intervel_test.items():
    info_ = dict_enroll_info_full[str(e_id)]
    info_.extend(static_)
    assenbled_data = info_

    list_test_static_info_dataset.append(assenbled_data)
    list_test_label.append(dict_test_label[int(e_id)])
    i+=1
    if (i%1000)==0:print(i)

json.dump(
    list_test_label,
    open('json_file\\into_model\\list_test_label.json','w'))
json.dump(
    list_test_static_info_dataset,
    open('json_file\\into_model\\list_test_static_info_dataset.json','w'))



dict_log_train = json.load(
    open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_log_train_sec_version.json'
        ,'r')
        ) 
dict_static_intervel_train = caculate_time_interval(dict_log_train)
  
i = 0
list_train_static_info_dataset = []
# c_inf0,u_info,long_static,short_static
list_train_label = []
for e_id,static_ in dict_static_intervel_train.items():
    info_ = dict_enroll_info_full[str(e_id)]
    info_.extend(static_)
    assenbled_data = info_

    list_train_static_info_dataset.append(assenbled_data)
    list_train_label.append(dict_train_label[int(e_id)])
    i+=1
    if (i%1000)==0:print(i)

json.dump(
    list_train_label,
    open('json_file\\into_model\\list_train_label.json','w'))
json.dump(
    list_train_static_info_dataset,
    open('json_file\\into_model\\list_train_static_info_dataset.json','w'))







for e_id , label in dict_train_label.items():
    list_info = dict_enroll_info[int(e_id)]

for e_id , label in dict_test_label.items():
    list_info = dict_enroll_info[int(e_id)]

