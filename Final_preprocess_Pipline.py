#%%
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
# 
from scipy import stats
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


def main(name:str,raw_folder_path:str):
  

    def preprocess(name,path):
        """[groupby enroll id and sort the log by time]

        Args:
            name ([type]): [description]
            path ([type]): [description]
        """    
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
                    
                    print('Start loading :',log_path)
                    log = pd.read_csv(
                        log_path
                        ,encoding=encoding_
                        ,names=columns)
                    print('Loading finish !')
                    
                
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

        def time_convert_and_sort(
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
        
        # load log file
        log_col = [
            'enroll_id',
            'username',
            'course_id',
            'session_id',
            'action',
            'object',
            'time'
            ]

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
        
        # sorted each log in dict by time

        dict_log_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log.json'
        hash_folder_path = 'hash_table_dict_file\\'
        path_eID_find_cID = hash_folder_path +name+'\\enroll_find_course.json'
        
        dict_log = json.load(open(dict_log_path,'r'))
        
        # drop time gap
        dict_log_after_time_convert_and_sort = time_convert_and_sort(
            dict_log,
            drop_zero = True,
            path_eID_find_cID= path_eID_find_cID )
        
        export_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log_ordered.json'
        json.dump(dict_log_after_time_convert_and_sort
            ,open(export_path,'w'))

        
        return dict_log_after_time_convert_and_sort

    def extract_feature(name:str,dict_log):
        """[info feature : dropout rate of users and courses]

        Args:
            name (str): [description]
            dict_log ([type]): [description]
        """        
        def assemble_info_data(name:str):
            """[concat user info and course info index by enroll id]

            Args:
                name (str): ['train' or 'test']
            Returns:
                    dict: [
                        e_id: 
                        [gender
                        ,birth_year
                        ,edu_degree
                        ,course_category
                        ,course_type
                        ,course_duration]
                        ]
            """    

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
                    u_id = int(enroll_find_user[e_id])
                    
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
                    return dict_enroll_info
                else: 
                    if  return_mode == 'list':
                        return    list_enroll_info

            c_info_path = 'raw_data_file\\course_info.csv'
            u_info_path = 'raw_data_file\\user_info.csv'
            u_info_col = ['user_id','gender','education_degree','birth_year']
            c_info_col = ['id','course_id','start_time','end_time','course_type','course_category']
            
            # load original info file
            U_INFO_NP = load(
                log_path =u_info_path,
                read_mode ='pandas',
                return_mode = 'values',
                encoding_ = 'utf-8',
                columns =u_info_col)
            C_INFO_NP = load(
                log_path =c_info_path,
                read_mode ='pandas',
                return_mode = 'values',
                encoding_ = 'utf-8',
                columns =c_info_col
                )

            dict_c_info = course_info_list_to_dict(C_INFO_NP[1:,:])
            dict_u_info = user_info_list_to_dict(U_INFO_NP[1:,:])

            # load hash table
            hash_path = 'hash_table_dict_file\\'+name
            enroll_find_course = json.load(
                open(hash_path+'\\enroll_find_course.json','r'))
            enroll_find_user   = json.load(
                open(hash_path+'\\enroll_find_user.json','r'))

            # eID is dict index
            info_dict_eID = make_info_dataset(
                return_mode = 'dict'
                ,dict_c_info  = dict_c_info
                ,dict_u_info  = dict_u_info
                ,enroll_find_course = enroll_find_course
                ,enroll_find_user   = enroll_find_user
                )

            return info_dict_eID

        def extract_feature_on_LogData(dict_log)-> dict:
            """[caculate static value of time interval,
                and 
                counting state transfor of log ]

            Args:
                dict_log ([type]): [description]

            Returns:
                dict: [
                    long_interval_static, # mean,var,skew,kurtosis
                    short_interval_static,# mean,var,skew,kurtosis
                    scene_transfer_count  # transfor matrix 4*4
                ]
            """    
            def static_(interval_list):
                    
                    if len(interval_list) > int(3):
                        R = interval_list
                        R_mean = np.mean(R) # 计算均值
                        R_var = np.var(R)   # 计算方差
                        R_skew = stats.skew(R)  #计算偏斜度 有偏
                        R_kurtosis = stats.kurtosis(R) #计算峰度 有偏

                        R_skew = np.abs(R_skew)
                        R_kurtosis = np.abs(R_kurtosis)

                        static_list = [
                            round(R_mean,2)
                            ,round(R_var,2)
                            ,round(R_skew,2)
                            ,round(R_kurtosis,2)]

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
            static_and_scene_dict = {}
            i = 0
            for e_id,list_log in dict_log.items():
                # 1 time interval
                long_interval_list  = []
                short_interval_list = []
                # 2 state transition
                scene_dict = {
                    '11':0,'12':0,'13':0,'14':0,
                    '21':0,'22':0,'23':0,'24':0,
                    '31':0,'32':0,'33':0,'34':0,
                    '41':0,'42':0,'43':0,'44':0 }
                for row in range(len(list_log)-1):
                    
                    row_next = list_log[row+1]
                    row_now  = list_log[row]

                    now_time = row_now[0]
                    

                    now_object = row_now[2]
                    now_session = row_now[3]
                    next_time = row_next[0]
                    
                    next_object = row_next[2]
                    next_session = row_next[3]
                    try:
                        time_interval = int(next_time - now_time)
                        if time_interval >int(60*15): # half hour 
                            long_interval_list.append(time_interval)
                        else:
                            short_interval_list.append(time_interval)
                    except:
                        print(next_time,now_time)
                    # scenes
                    # 424642151213
                    # 444444
                    # 121113
                    if row < (len(list_log)-2):
                        # row_now  = list_log[row]
                        now_action = row_now[1]
                        next_action = list_log[row+1][1]
                        #nextnext_action = list_log[row+2][1]

                        a0 = str(now_action)[0]
                        a1 = str(next_action)[0]
                        #a2 = str(nextnext_action)[0]

                        scene_ = a0+a1

                        scene_dict[scene_]+=1
                        
                
                short_static = static_(short_interval_list)
                long_static  = static_(long_interval_list)
                scene_list = list(scene_dict.values())
                # 3 head/tail gap
                    
                i+=1
                if (i%1000)==0:print(i)

                # time_interval_dict[int(e_id)] = interval_list
                long_static.extend(short_static)
                static_list = long_static
                # static_interval_dict[int(e_id)] = static_list
                # enroll_scene_dict[int(e_id)] = scene_list
                static_list.extend(scene_list)
                static_and_scene_list = static_list
                static_and_scene_dict[int(e_id)] = static_and_scene_list
                # break
            return static_and_scene_dict #static_interval_dict,enroll_scene_dict
        
        def extract_feature_on_InfomationData(
            mode: str,
            threshold_course_amount = int(3),
            threshold_student_amount = int(3)):
            """[caculate_droupout_rate]

            Returns:
                [type]: [description]
            """            
            # prepare for hot data
            # unsuitable for cold start
            # need thershod to choose enable or not
            # 训练结果可以关联到课程分类上
            def load(
                log_path: str,
                return_mode: str,
                encoding_='utf-8',
                read_mode = 'pd',
                columns=None,
                train=train_OR_NOT)-> ndarray or DataFrame:
                '''读取csv文件 返回numpy数组'''
            
                if read_mode == 'pd' :
                    import pandas as pd
                    if train ==True: # only read 10000rows 
                        reader = pd.read_csv(
                            log_path
                            ,encoding=encoding_
                            ,names=columns
                            ,chunksize=chunk_size)
                            
                        for chunk in reader:
                            # use chunk_size to choose the size of train rows instead of loop
                            log = chunk
                            return log.values

                    else: # read full file
                        print('Start loading ',log_path)
                        log = pd.read_csv(
                            log_path
                            ,encoding=encoding_
                            ,names=columns)
                        print('Loading finsih!')
                    
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
            
            def dict_cID_and_uID_find_label(
                enroll_find_course,
                enroll_find_user,
                dict_enroll_label)->dict:
                new_dict = {}
                for e_id in enroll_find_course.keys():
                    
                    e_id = str(e_id)
                    c_id = enroll_find_course[e_id]
                    u_id = enroll_find_user[e_id]


                    new_key = str(u_id)+c_id # str + str
                    label   = dict_enroll_label[int(e_id)]
                    new_dict[new_key] = label
                return new_dict
            # load hash table
            hash_path = 'hash_table_dict_file\\'+mode
            
            user_find_course = json.load(
                open(hash_path+'\\user_find_course.json','r'))
            course_find_user = json.load(
                open(hash_path+'\\course_find_user.json','r'))
            
            enroll_find_course = json.load(
                open(hash_path+'\\enroll_find_course.json','r'))
            enroll_find_user   = json.load(
                open(hash_path+'\\enroll_find_user.json','r'))

            if mode == 'train':
                enroll_find_label = list_to_dict(
                    load(
                        log_path='raw_data_file\\'+name+'_truth.csv'
                            # columns=['e_id','drop_or_not']
                        ,return_mode = 'list')
                        )
                u_and_c_find_label = dict_cID_and_uID_find_label(
                    enroll_find_course = enroll_find_course,
                    enroll_find_user   = enroll_find_user,
                    dict_enroll_label  = enroll_find_label )
                # users means user's
                users_courseAmount_and_drop_rate = {}
                for u_id,courses in user_find_course.items():
                    label_list = []
                    for c_id in courses:
                        key = str(u_id)+str(c_id)
                        label = u_and_c_find_label[key]
                        label_list.append(label)
                    
                    course_amount = len(courses)
                    
                    if course_amount < int(threshold_course_amount):
                        dropout_rate = np.nan()
                    else:
                        dropout_rate = int((
                            sum(label_list)/len(label_list)
                            )*100)
                    
                    users_courseAmount_and_drop_rate[u_id] = [
                        course_amount,
                        dropout_rate ]
                    
                    #json.dump(users_courseAmount_and_drop_rate,open('json_file\\info_2.0\\users_courseAmount_and_drop_rate.json','w'))
                # courses means course's
                courses_studentAmount_and_drop_rate = {}
                for c_id,users in course_find_user.items():
                    label_list = []
                    for u_id in users:
                        key = str(u_id)+str(c_id)
                        label = u_and_c_find_label[key]
                        label_list.append(label)
                    
                    user_amount = len(users)
                    if user_amount < threshold_student_amount:
                        dropout_rate = np.nan()
                    else:
                        dropout_rate = int((
                            sum(label_list)/len(label_list)
                            )*100)
                    courses_studentAmount_and_drop_rate[c_id] = [
                        user_amount,
                        dropout_rate ]
                
                # Extract Finish

                # Export 
                # {u_id:[course_amount,drop_rate]}
                json.dump(
                    users_courseAmount_and_drop_rate,
                    open('extracted_features\\users_courseAmount_and_drop_rate.json','w'))
                
                # {c_id:[student_amount,drop_rate]}
                json.dump(
                    courses_studentAmount_and_drop_rate,
                    open('extracted_features\\courses_studentAmount_and_drop_rate.json','w'))

            if mode== 'test':
                
                # {u_id:[course_amount,drop_rate]}
                users_courseAmount_and_drop_rate    = json.load(
                    open('extracted_features\\user_dropout_rate.json','r'))
                
                # {c_id:[student_amount,drop_rate]}
                courses_studentAmount_and_drop_rate = json.load(
                    open('extracted_features\\courses_studentAmount_and_drop_rate.json','r'))
                
            # Assemble
            e_id_list = list(enroll_find_course.keys())
            info_feature_dict = {}

            for e_id in e_id_list:

                e_id = int(e_id)
                c_id = enroll_find_course[e_id]
                u_id = enroll_find_user[e_id]
               
                try:
                    student_amount         = courses_studentAmount_and_drop_rate[c_id][0],
                except:
                    student_amount = np.nan()
                try:
                    course_amount          = users_courseAmount_and_drop_rate[u_id][0],
                except:
                    course_amount = np.nan()
                try:
                    dropout_rate_of_course = courses_studentAmount_and_drop_rate[c_id][1],
                except:
                    dropout_rate_of_course = np.nan()
                try:
                    dropout_rate_of_user   = users_courseAmount_and_drop_rate[u_id][1]
                except:
                    dropout_rate_of_user = np.nan()

                info_feature_dict[e_id] = [
                     student_amount,
                     course_amount,
                     dropout_rate_of_course,
                     dropout_rate_of_user
                    ]
            
            return info_feature_dict

            


        if name == 'train':
            
                info_dict         = assemble_info_data(name = 'train')
                log_feature_dict  = extract_feature_on_LogData(dict_log)
                info_feature_dict = extract_feature_on_InfomationData(mode = 'train')
                
                # 组合以上三个 召唤神龙

                
        return log_feature_dict

      
    

    data_list  = None
    label_list = None

     
    if name == 'test':
        pass
    
    if name == 'train':
        
        dict_train_log = preprocess(
            name='train', 
            path = raw_folder_path+'train_log.csv')

        dict_train_feature = extract_feature(
            name = 'train',
            dict_log = dict_train_log
            )
         
        # HERE need return list type not dict type


    

    
    return data_list,label_list


list_test_data,list_test_label = main(
    name = 'test',
    raw_folder_path = 'raw_data_file\\')

list_train_data,list_train_label = main(
    name = 'train',
    raw_file_folder_path = 'raw_data_file\\')


