# type
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
# 
import pandas as pd
import numpy as np
import json

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


def to_dict_2(
    log: ndarray
    ,mode: str
    ,test=TEST_OR_NOT
    )-> Dict[str,ndarray]: 

    """[groupby enrollment number and encoding the feature]

    Returns:
        [type:dict]: 
        { enroll_id:   
            data = [
                action_time,
                action,        # int
                action_object, # int
                session        # int]}
    """   
    import numpy as np
    print('find ',len(log),' row logs')
    i = 0
    log_dict = {}

    # query dict
  #  user_find_course = {}
  #  course_find_user = {}

    enroll_find_user = {}
    enroll_find_course = {}

  #  course_find_enroll = {}
  #  user_find_enroll = {}

    

    # replace 
    action_replace_dict = {
        # video
        'seek_video': 11

        ,'load_video':12
        ,'play_video':12

        ,'pause_video':14
        ,'stop_video':14
        
        # problem
        ,'problem_get':21
        ,'problem_check':21
        ,'problem_save':21

        ,'reset_problem':24
        ,'problem_check_correct':25
        , 'problem_check_incorrect':26
        # comment
        ,'create_thread':31
        ,'create_comment':32
        ,'delete_thread':33
        ,'delete_comment':34
        # click
        ,'click_info':41
        ,'click_courseware':42
        ,'close_courseware':42
        ,'click_about':43
        ,'click_forum':44
        ,'close_forum':44
        ,'click_progress':45
       
       }
    object_replace_dict = {}
    session_replace_dict = {}

    object_count = 1
    session_count = 1

    for row in log:
        # id
      

        enroll_id = row[0]
        user_id = row[1]
        course_id = row[2]
        # feature
        
        if row[3] is np.nan :session= 0
        else:session = row[3]
        if row[4] is np.nan :action= 0
        else:action = row[4]
        if row[5] is np.nan :action_object= 0
        else:action_object = row[5]
        
       # try:    
        action_time = row[6]

        #print('query')
        #id query dict
      #  user_find_course[user_id] = course_id
       # course_find_user[course_id] = user_id

        enroll_find_course[enroll_id] = course_id
        course_find_enroll[course_id] = enroll_id

        user_find_enroll[user_id] = enroll_id
        enroll_find_user[enroll_id] = user_id
        
       
        # int replace str
        try:
            action = action_replace_dict[action]
        except:
            action = int(0)
        
        try:
            action_object = object_replace_dict[action_object]
        except:
            object_count +=1
            object_replace_dict[action_object] = object_count
            action_object = object_replace_dict[action_object]
        
        try:
            session = session_replace_dict[session]
        except:
            session_count +=1
            session_replace_dict[session] = session_count
            session = session_replace_dict[session]

        data = [
            action_time,
            action,
            action_object,
            session ]
        # if log_dict[]里没数据：初始化=[]
        try:
            log_dict[enroll_id].append(data)
        except:
            log_dict[enroll_id] = []
            log_dict[enroll_id].append(data)
            #print(log_dict[enroll_id])
        i+=1
        if (i%print_batch)==0:print('already dict : ',i,'row logs')

   
    def export_json_file():
        import json
        json.dump(action_replace_dict,open('json_file\\'+str(mode)+'_dataset\\action_replace_dict.json','w'))
        json.dump(object_replace_dict,open('json_file\\test_dataset\\object_replace_dict.json','w'))

      #  json.dump(user_find_course,open('json_file\\'+str(mode)+'_dataset\\user_find_course.json','w'))
      #  json.dump(course_find_user,open('json_file\\'+str(mode)+'_dataset\\course_find_user.json','w'))

        json.dump(enroll_find_user,open('json_file\\'+str(mode)+'_dataset\\enroll_find_user.json','w'))
        json.dump(enroll_find_course,open('json_file\\'+str(mode)+'_dataset\\enroll_find_course.json','w'))
        
        json.dump(course_find_enroll,open('json_file\\'+str(mode)+'_dataset\\course_find_enroll.json','w'))
        json.dump(user_find_enroll,open('json_file\\'+str(mode)+'_dataset\\user_find_enroll.json','w'))
    
    export_json_file()

    if (test == True) and (i ==print_batch):
        return log_dict
    else:
        json.dump(
            log_dict,
            open(
                'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_'+str(mode)+'_log.json'
                ,'w'))
        return log_dict



def convert(
    log: dict 
    ,drop_zero: bool
   
    )->dict: 
    """[summary]

    Args:
        log (dict): [description]
        drop_zero (bool): [description]

    Returns:
        [type:dict]:
            {
                enroll_id:
                    data = [
                         action_time
                         , action
                         , action_object
                         , session 
                         ]
            }
    """    
    print('dict total len :',len(log))
    print(' convert running!')
    import json
    import numpy as np
    dict_enrollID_find_courseID = json.load(open(path_eID_find_cID,'r'))

    def find_start_end(e_id:str)->Dict[int,datetime64]:

        ''' 根据course_id 查询课程的总耗时秒数 以及开始时间并返回
            函数调用了全局变量C_INFO_NP必须在课程信息被加载后才能运行'''
        c_id = dict_enrollID_find_courseID[str(e_id)]
        mask = C_INFO_NP[:,1] == c_id

        start = C_INFO_NP[mask][:,2]
        end   = C_INFO_NP[mask][:,3]
        #type: object ['2016-11-16 08:00:00']
        start = str(start)
        end = str(end)
        #type: str ['2016-11-16 08:00:00']
        start = start[2:-2]
        end = end[2:-2]
        #type: str '2016-11-16 08:00:00'
        try:
            end = np.datetime64(end) 
            start = np.datetime64(start)
            seconds_of_gap = int((end - start).item().total_seconds())
        except:print('ERROR start,end :',start,end)
        time_info = {
            'length': seconds_of_gap
            ,'head' : start}
        return time_info
    
    def time_map(log_np:ndarray)->ndarray:
        ''' 本函数功能：
                    将以秒数为索引的不确定顺序nparray 
                    映射到按秒数排序的ndarray 
                    返回值近保留有序的action序列
            BUG日志：
                    20210113pm 
                    原始数据中存在错误的时间格式
                    本map函数遇到错误格式直接忽略本循环
                    会导致错误行的action值为 b''
                    进而导致int（）转换出错
                    报错：ValueError: invalid literal for int() 
                            with base 10: ''
                    解决方案： 在字符替换表中先判断若为b'' 则先替换为b'0'
                     '''
        '''action_series改成无零的action有序表 '''
        
        ''' def to_int(x):
                x = int(x)
                return x
            md = map(to_int,log_np[:,1]) 
            __time = list(md) # time list'''
       
        '''    time_column = log_np[:,1]
            for __row in range(len(time_column)):
                try:
                    time_column[__row] = int(time_column[__row])
                except: 
                    print(' e_id in log :',e_id,'row number :',__row)

            __time = time_column'''

        __time = log_np[:,0].astype('int')

        __head = np.min(__time)
        __tail = np.max(__time)
        __length = __tail - __head +1
        action_series = np.zeros((__length,3),dtype=np.uint32)

        for row in log_np:
            __t = int(row[0]) # time now
            __location = __t - __head
            action_series[__location,:] = row[1:]
        if drop_zero == True:
            mask = action_series!= np.uint8(0)
            action_series = action_series[mask]

        return action_series
    
    i = 0
    new_dict = {}
    for e_id ,v in log.items():
        i+=1
       
        if (i%int(1000))==0:
            print('already convert ',i,' e_id ')
  
        # type(v)==list
        v = np.array(v)
        _log       = v[:,[1,2,3]]
        time_col   = v[:,0]

        # action_col = _log[:,1] 
        # object_col = _log[:,2]
        # session_col = _log[:,3]

        time_info   = find_start_end(str(e_id))
        time_head   = time_info['head']
        time_length = time_info['length']

        np_time   = np.zeros((len(_log),1),dtype = np.uint32)
        
        np_feature = np.array(_log,dtype = np.uint32) # a32 存32个英文字符
        

        for row_num in range(len(_log)):
           
            _row = _log[row_num,:]
            _time = time_col[row_num]
           
            try:
                _time = np.datetime64(_time)
                _time =  int(
                    ( _time - time_head ).item().total_seconds() )
                
                np_time[row_num] = _time
                

            except:
                print('ERROR log time [_time] :',_time)
                
                print('np_time :',np_time,'np_feature :',np_feature)
                
        
        rebulid = np.concatenate( ( np_time ,np_feature ), axis = 1)
        rebulid = rebulid[ rebulid[:,0].argsort()]
        '''出于保留 ‘用户主要的操作分布在开课时间的哪一部分’ 这一特征 
            的目的，将时间转换部分分为两部分写，后期如需重建此特征以上的代码可以不动'''

        action_series = rebulid # time_map(rebulid)
        new_dict[e_id] =  action_series
        
        
    return new_dict


# course infomation file
c_info_path = 'course_info.csv'
c_info_col = [
    'id',
    'course_id',
    'start','end',
    'course_type',
    'category']
C_INFO_NP = load(
    log_path =c_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =c_info_col
    )



log_path = {
    'test':'D:\\zyh\\data\\prediction_data\\prediction_log\\test_log.csv',
    'train':'D:\\zyh\\data\\prediction_data\\prediction_log\\train_log.csv' }

# columns in log files
log_col = [
    'enroll_id',
    'username',
    'course_id',
    'session_id',
    'action',
    'object',
    'time']

for name,path in log_path.items():
    # log_np: log is numpy.ndarray data type
    log_np = load(
        log_path =path,
        read_mode ='pandas',
        return_mode = 'values', # 'values': ndarray , 'df': dataframe
        encoding_ = 'utf-8', 
        columns =log_col)

    log_dict = to_dict_2( log_np[1:,:])  # column 0 is column index 
    # columns : e_id , action , time , c_id


# test data
dict_log_test = json.load(
    open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_test_log.json'
        ,'r')
        )
path_eID_find_cID = 'json_file\\test_dataset\\enroll_find_course.json'
dict_log_test_sec_version = convert(dict_log_test,drop_zero = True) # drop time gap
for k,v in dict_log_test_sec_version.items():
    t[int(k)] = v.tolist()
json.dump(
    dict_log_test_sec_version
    ,open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_log_test_sec_version.json'
        ,'w'))

# train data
dict_log_train = json.load(
    open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_train_log.json'
        ,'r')
        )
path_eID_find_cID = 'json_file\\train_dataset\\enroll_find_course.json'
dict_log_train_sec_version = convert(dict_log_train,drop_zero = True) # drop time gap
t = {}
for k,v in dict_log_train_sec_version.items():
    t[int(k)] = v.tolist()
dict_log_train_sec_version = t
json.dump(
    dict_log_train_sec_version
    ,open(
        'D:\\zyh\\data\\prediction_data\\advance_preprocess_json\\dict_log_train_sec_version.json'
        ,'w'))

