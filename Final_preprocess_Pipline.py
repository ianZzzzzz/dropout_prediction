# type
#%%


from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
# 
import pandas as pd
import numpy as np
import json

#%%
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


def log_groupby_to_dict(
    log: ndarray or list
    ,mode: str # 'train' or 'test' 
    ,test=TEST_OR_NOT
    )-> Dict[int,list]: 
    # origin name to_dict_2
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
    
    print(str(mode)+' log amount :',len(log),' rows')
    i = 0
    log_dict = {}

    # hash table dict
    user_find_course = {}
    user_find_enroll = {}
    
    enroll_find_user = {}
    enroll_find_course = {}

    course_find_enroll = {}
    course_find_user = {}


    

    # Encoding 

    # keyword repalce dict 

    # actions is fixed 
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
    
    # objects and sessions is  dynamically add to dict
    object_replace_dict  = {}
    session_replace_dict = {}

    object_count  = 0
    session_count = 0

    for row in log:

        # id
        enroll_id = int(row[0])
        user_id   = row[1]
        course_id = row[2]

        # feature
        if row[3] is np.nan :
            session  = int(0)
        else:session = row[3]

        if row[4] is np.nan :
            action  = int(0)
        else:action = row[4]

        if row[5] is np.nan :
            action_object  = int(0)
        else:action_object = row[5]
        
       # try:    
        action_time = row[6]


        # Making id hash dict
        user_find_course[user_id]   = course_id
        course_find_user[course_id] = user_id

        enroll_find_course[enroll_id] = course_id
        course_find_enroll[course_id] = enroll_id

        user_find_enroll[user_id]   = enroll_id
        enroll_find_user[enroll_id] = user_id
        
       
        # int replace str
        try:
            action = action_replace_dict[action]
        except:
            action = int(0)
        
        try:
            action_object = object_replace_dict[action_object]
        except:
            # the number of object and session now is unknow
            # hence , caculate the amount of objects  
            # and replace str by the number of object
            object_count +=1
            object_replace_dict[action_object] = object_count
            action_object = object_replace_dict[action_object]
        
        try:
            session = session_replace_dict[session]
        except:
            # the number of object and session now is unknow
            # hence , caculate the amount of sessions  
            # and replace str by the number of session
            session_count +=1
            session_replace_dict[session] = session_count
            session = session_replace_dict[session]

        data = [
            action_time,
            action,
            action_object,
            session ]

        
        # if log_dict[] is empty -> init = []
        try:
            log_dict[enroll_id].append(data)
        except:
            log_dict[enroll_id] = []
            log_dict[enroll_id].append(data)
            #print(log_dict[enroll_id])
        i+=1
        if (i%print_batch)==0:print('already processed : ',i,'row logs')

   
    hash_tables_dict = {
        'action_replace_dict' :action_replace_dict,
        'object_replace_dict' :object_replace_dict,
        'session_replace_dict':session_replace_dict,

        'course_find_user'  :course_find_user,
        'course_find_enroll':course_find_enroll,
        'enroll_find_user'  :enroll_find_user,
        'enroll_find_course':enroll_find_course,
        'user_find_course'  :user_find_course,
        'user_find_enroll'  :user_find_enroll 
        }
    hash_folder = 'hash_table_dict_file\\'

    for name,data in hash_tables_dict.items():
        path = hash_folder +str(mode)+'\\'+name +'.json'
        json.dump(data,open(path,'w'))
        print('already dump ',name,'.json to ',path)


    if (test == True) and (i ==print_batch):
        return log_dict
    else:
        json.dump(
            log_dict,
            open('after_processed_data_file'+'\\'+str(mode)+'\\dict_'+str(mode)+'_log.json','w'))
        return log_dict



def time_convert(
    log: dict 
    ,drop_zero: bool
    ,path_eID_find_cID: str
    )->Dict[int,list]: 
    """[summary]
        Origin time format : str , un-ordered
        After this function: int , ordered
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
        # un-used 
        # reason ： cost more memory then argsort 
        # benefit : faster then argsort
        ''' [dercribe]：
                    sort the time ndarray in  k* n complexity
                    k*n have best time efficiency but un-stable memory cost
                    
                    给定时间起始点与总长度
                    列表无序存储了区间内任意个时间点
                    对列表进行排序
                    
                    分配与总长度相同的列表空间
                    对每一个时间点数据减去时间起始点
                    将差值作为索引存入列表空间
                    即得到有序列表
                    
            BUG log：
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

        np_time   = np.zeros(
            (len(_log),1) ,dtype = np.uint32)
        
        np_feature = np.array(_log,dtype = np.uint32) 
        

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


        new_dict[int(e_id)] =  rebulid.tolist()
        
        
    return new_dict

#%%
# course infomation file
c_info_path = 'raw_data_file\\course_info.csv'
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


#%%
log_path = {
    'test':'raw_data_file\\test_log_t.csv',
    'train':'raw_data_file\\train_log_t.csv' }

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

    log_dict = log_groupby_to_dict( 
        log_np[1:,:],
        mode = name)  
    # column 0 is column index 
    # columns : e_id , action , time , c_id

for name in ['test','train']:
    dict_log_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log.json'
    hash_folder_path = 'hash_table_dict_file\\'
    path_eID_find_cID = hash_folder_path +name+'\\enroll_find_course.json'
    
    dict_log = json.load(open(dict_log_path,'r'))
    
    # drop time gap
    dict_log_after_time_convert_and_sort = time_convert(
        dict_log,
        drop_zero = True,
        path_eID_find_cID= path_eID_find_cID )
    
    export_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log_ordered.json'
    json.dump(dict_log_after_time_convert_and_sort
        ,open(export_path,'w'))


#%%