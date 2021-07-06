
#%%

from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import matplotlib.pyplot as plt

import json
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

import xgboost as xgb
from sklearn import linear_model   
#%%
def transfor_matrix_process(
    name:str,
    raw_folder_path:str,
    transfor_matrix:str,
    analy_mode = True,
    load_log_from_json = True,
    export = False):
    TEST_OR_NOT = False
    print_batch = int(1000000)
    chunk_size  = int(10000) # enable only when TEST_OR_NOT = True
    from scipy import stats
    import pandas as pd
    import numpy as np
    import json

    def load_label(mode:str,return_mode = 'list')->list:
        def load(
            log_path: str,
            return_mode='values',
            read_mode='pandas',
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
                    
                    print('    Start loading :',log_path)
                    log = pd.read_csv(
                        log_path
                        ,encoding=encoding_
                        ,names=columns)
                    print('      Total length : ',len(log),'rows.')
                    
                
            if return_mode == 'df':return log
            if return_mode == 'values':return log.values

        print('  load_label running : ')
        np_label = load(
            log_path = raw_folder_path+mode+'_truth.csv')
        
        if return_mode == 'list':
            print('    return list label.\n')
            print('  load_label finish.\n')
            return np_label[:,1].tolist()
        
        if return_mode == 'dict':
            def list_to_dict(
                list_:list,
                key_type  = 'int',
                value_type='int')-> dict:
                """[convert dict to list 
                use the 1st cloumn make index 2nd column make value]

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
            
            dict_label = list_to_dict(list_ = np_label.tolist())
            print('    return dict label.\n')
            print('  load_label finish.\n')
            return dict_label
            
    def to_df(
        sample:list,
        label: list,
        e_id_list)->DataFrame:
        
        df_data = pd.DataFrame(
            data=sample,
            columns=[
                'L_mean','L_var','L_skew','L_kurtosis',
                'S_mean','S_var','S_skew','S_kurtosis',
                'video-video','video-answer','video-comment','video-courseware',
                'answer-video','answer—answer','answer-comment','answer-courseware',
                'comment-video','comment-answer','comment-comment','comment-courseware',
                'courseware-video','courseware-answer','courseware-comment','courseware-courseware',
                
                'gender','birth_year' ,'edu_degree',
                'course_category','course_type','course_duration',
                'course_amount','dropout rate of course',
                'student_amount',' dropout rate of user']
            )

        df_label = pd.DataFrame(
            data=label,
            columns=['drop_or_not'])
        df_e_id = pd.DataFrame(
            data = e_id_list,
            columns= ['enroll_id'])
        return df_data,df_label,df_e_id

    def preprocess(name,path):
        """[groupby enroll id and sort the log by time]

        Args:
            name ([type]): [description]
            path ([type]): [description]
        """    
        def load(
            log_path: str,
            return_mode: str,
            read_mode='pandas',
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
                    
                    print('    Loading :',log_path)
                    log = pd.read_csv(
                        log_path
                        ,encoding=encoding_
                        ,names=columns)
                    print('    Total length :',len(log),'rows.')
                    
                
            if return_mode == 'df':return log
            if return_mode == 'values':return log.values

        def log_groupby_to_dict(
            log: ndarray or list
            ,mode: str # 'train' or 'test' 
            ,test=TEST_OR_NOT
            )-> Dict[int,list]: 
            # predicted name to_dict_2
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
            print('\n    Log_groupby_to_dict running : \n')
            print('      ',str(mode)+' log amount :',len(log),' rows')
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
                user_id   = int(row[1])
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
                

                action_time = row[6]


                # Making id hash dict
                
                '''
                def make_hash_table(main_key,sub_key,hash_table):
                    try:
                        if sub_key not in hash_table[main_key]:
                            hash_table[main_key].append(sub_key)
            
                    except:
                        hash_table[main_key] = [sub_key]
                '''
                
                # user_find_course[user_id].append(course_id) 
                try:
                    if course_id not in user_find_course[user_id]:
                        user_find_course[user_id].append(course_id)
        
                except:
                    user_find_course[user_id] = [course_id]

                # course_find_user[course_id].append(user_id) 
                try:
                    if user_id not in course_find_user[course_id]:
                        course_find_user[course_id].append(user_id)
        
                except:
                    course_find_user[course_id] = [user_id]

                enroll_find_course[enroll_id] = course_id
                '''try:
                    if course_id not in enroll_find_course[enroll_id]:
                        enroll_find_course[enroll_id].append(course_id)
        
                except:
                    enroll_find_course[enroll_id] = [course_id]
                '''
                # course_find_enroll[course_id].append(enroll_id)
                try:
                    if enroll_id not in course_find_enroll[course_id]:
                        course_find_enroll[course_id].append(enroll_id)
        
                except:
                    course_find_enroll[course_id] = [enroll_id]
                # user_find_enroll[user_id].append(enroll_id)
                try:
                    if enroll_id not in user_find_enroll[user_id]:
                        user_find_enroll[user_id].append(enroll_id)
        
                except:
                    user_find_enroll[user_id] = [enroll_id]

                enroll_find_user[enroll_id] = user_id
                '''try:
                    if user_id not in enroll_find_user[enroll_id]:
                        enroll_find_user[enroll_id].append(user_id)
        
                except:
                    enroll_find_user[enroll_id] = [user_id]
                '''
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

            print('    log_groupby_to_dict finish. ')
            print('\n    export hash tables running : \n')
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
                if export == True:
                    json.dump(data,open(path,'w'))
            print('      export hash tables finish.')


            if (test == True) and (i ==print_batch):
                return log_dict
            else:
                
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
            print('\n    time_convert_and_sort running : ')
            print('    Total action series:',len(log))
          
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
                
            print('    time_convert_and_sort finish. \n')  
            return new_dict

        print('\n  Preprocess running : \n')    
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

        dict_log = log_groupby_to_dict( 
            log_np[1:,:],
            mode = name)  
        log_np = None # release ram

        # column 0 is column index 
        # columns : e_id , action , time , c_id
        
        # sorted each log in dict by time

        # dict_log_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log.json'
        hash_folder_path = 'hash_table_dict_file\\'
        path_eID_find_cID = hash_folder_path +name+'\\enroll_find_course.json'
        
        # print('    Loading dict_log.')
        # dict_log = json.load(open(dict_log_path,'r'))
        # print('    Success load dict_log.')
        
        # drop time gap
        dict_log_after_time_convert_and_sort = time_convert_and_sort(
            dict_log,
            drop_zero = True,
            path_eID_find_cID= path_eID_find_cID )
        dict_log = None
        print('    Exproting processed dict_log.')
        export_path = 'dict_log'+'\\dict_'+name+'_log_ordered.json'
        if export == True:
            json.dump(dict_log_after_time_convert_and_sort
                ,open(export_path,'w'))
        print('    Export finish , path :',export_path)
        print('\n  Preprocess finish. \n')
        return dict_log_after_time_convert_and_sort


    def extract_feature(
  
            name:str,
            dict_log,
            return_mode = 'dict')-> dict:

        def extract_feature_on_LogData(
            dict_log,
            )-> dict:

            print("\n    extract_feature_on_LogData running : \n")
            time_interval_dict   = {}
            
            dict_eid_scene = {}
            i = 0
            # init scene dict
            if transfor_matrix=='simple':
                scene_dict = {
                    '11':0,'12':0,'13':0,'14':0,
                    '21':0,'22':0,'23':0,'24':0,
                    '31':0,'32':0,'33':0,'34':0,
                    '41':0,'42':0,'43':0,'44':0 }
            if transfor_matrix=='complex':
                
                video_      = ['11','12','13','14','15']
                problem_    = ['21','22','23','24','25','26']
                comment_    = ['31','32','33','34']
                courseware_ = ['41','42','43','44','45','46','47']
                scene_dict    = {}
                for start in [*video_,*problem_,*comment_,*courseware_]:

                    for end in [*video_,*problem_,*comment_,*courseware_]:
                        scene = str(start)+str(end)
                        scene_dict[ scene ] = 0

            for e_id,list_log in dict_log.items():
        
                # 2 state transition
                
                for row in range(len(list_log)-1):
                    
                    row_next = list_log[row+1]
                    row_now  = list_log[row]

                    if row < (len(list_log)-2):
                        # row_now  = list_log[row]
                        now_action = row_now[1]
                        next_action = row_next[1]
                        #nextnext_action = list_log[row+2][1]
                        if transfor_matrix =='simple':
                                
                            a0 = str(now_action)[0]
                            a1 = str(next_action)[0]
                            #a2 = str(nextnext_action)[0]
                        if transfor_matrix =='complex':
                            a0 = str(now_action)
                            a1 = str(next_action)
                            
                        scene_ = a0+a1

                        scene_dict[scene_]+=1
                        
                
            
                scene_list = list(scene_dict.values())

                # 3 head/tail gap
                    
                i+=1
                if (i%20000)==0:
                    print('already processed  ',i,' enrollment id')

                dict_eid_scene[int(e_id)] = scene_list
                # break
            return dict_eid_scene

        
        log_feature_dict  = extract_feature_on_LogData( dict_log)
        
        return log_feature_dict

    if load_log_from_json == True:
        dict_log_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log_ordered.json'
        dict_log = json.load(open(dict_log_path,'r'))
    else:
        dict_log = preprocess(
            name=name, 
            path = raw_folder_path+name+'_log.csv')
    
    #json.dump(dict_log,open('after_processed_data_file//'+name+'_dict.json','w'))
    
    dict_data = extract_feature(

         name = name,
        dict_log = dict_log
        )
    dict_log = None # release memory


    if analy_mode ==True :
    
        return dict_data
    else:
        list_data  = []
        list_label = []
        list_e_id  = []

        for e_id,dict_ in dict_data.items():
            list_data.append(dict_['features'])
            list_label.append(dict_['label'])
            list_e_id.append(int(e_id))

            if export == True:
                json.dump(
                    list_data,
                    open('Final_Dataset\\'+name+'_data.json','w'))
                json.dump(
                    list_label,
                    open('Final_Dataset_T\\'+name+'_label.json','w'))
                print('Major Data Process is finish.')     
            

    
        return to_df(list_data,list_label,e_id_list=list_e_id)
def choose_data_from_dict(mode:str)-> dict:
    path = 'after_processed_data_file\\dict_data_'+mode+'_for_analy_fix_birth.json'
    dict_ = json.load(open(path,'r'))
    new_dict = {}
    for eid,dict__ in dict_.items():
        static = dict__['log_features'][:8]
        infomation = dict__['info_features']
        label = dict__['label']

        new_dict[eid] = {}
        new_dict[eid]['static'] = static
        new_dict[eid]['infomation'] = infomation
        new_dict[eid]['label'] = label

    return new_dict
def assemble_transfor_matrix_and_others(matrix_dict,others_dict)-> dict:
    if len(matrix_dict)!=len(others_dict):print('Error length mismatch.')
    else:
        new_dict = {}
        for eid in others_dict.keys():

            new_dict[eid] = {}
            new_dict[eid]['features'] = [
                                    *others_dict[eid]['static'] ,
                                    *others_dict[eid]['infomation'] ,
                                    *matrix_dict[int(eid)] ]
            new_dict[eid]['label'] = others_dict[eid]['label']

        return new_dict
def prepare_label(dict_data)-> list:
    """
        dict_data : { e_id: {
                            'log_features':[] ,
                            'info_features':[] ,
                            'label' : 1 or 0
                            } 
                    }

    Returns:
        list_assemble_data : [[ *log_features , *info_features ],......]
        list_e_id : [ e_id_1 ,e_id_2,...... ]
        list_label : [ 0, 1, ..........]
    """    
    list_assemble_data = []
    list_e_id = []
    list_label = []
    for e_id,dict_ in dict_data.items():

        log_data  = dict_['features']

        label     = dict_['label']

        list_assemble_data.append(log_data)
        list_e_id.append(int(e_id))
        list_label.append(int(label))

    return list_assemble_data , list_e_id , list_label
def nomilized_matrix(dict_matrix,matrix_type: str,ignore=False)-> dict:
    import numpy as np
    if matrix_type =='complex':
        for eid,dict_ in dict_matrix.items():
            matrix_= dict_['features'][18:]
            sum_ = np.sum(matrix_)
            if (sum_ != 0):
                for i in range(18,502):
                    dict_matrix[eid]['features'][i] =int(30000*dict_matrix[eid]['features'][i]/sum_)
        return dict_matrix
    if matrix_type =='simple':
        new_dict = {}
        for eid,dict_ in dict_matrix.items():
            if ignore ==False:    
                matrix_= dict_['log_features'][8:]           
                sum_ = np.sum(matrix_)
                if sum_ != 0:
                    for i in range(8,24):
                        try:
                            dict_matrix[eid]['log_features'][i] =int(1000*dict_matrix[eid]['log_features'][i]/sum_)
                        except:
                            print(dict_matrix[eid]['log_features'][i],sum_)
                        
                    
            new_dict[eid]={}
            new_dict[eid]['features'] = [
                *dict_matrix[eid]['info_features'],
                *dict_matrix[eid]['log_features']
                ]
           
            new_dict[eid]['label'] = dict_matrix[eid]['label']

        return new_dict
def predict_label_to_int(predict_label_list,threshold):

    predict_label_int = []
    for i in predict_label_list:
        value= i
        if value >threshold:label_ = int(1)
        else:label_ = int(0)
        predict_label_int.append(label_)

    return predict_label_int

def measure(predict_label_int,list_label_test):
    f1             = f1_score(predict_label_int,list_label_test)
   # accuracy = accuracy_score(predict_label_int,list_label_test)
   # AUC =       roc_auc_score(predict_label_int,list_label_test)
    # precision = precision_score(predict_label_int,test_label.tolist())
    # recall = recall_score(predict_label_int,test_label.tolist())

    print('F1',round(f1,4))
def plot_AUC(ori_label,predict_label):
    import pylab as plt
    import warnings;warnings.filterwarnings('ignore')
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(ori_label, predict_label)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
def xgb_model(train_data,test_data)->list:
    dtrain = xgb.DMatrix(train_data, list_label_train)
    dtest = xgb.DMatrix(test_data,list_label_test)
    num_rounds = 100
    params = {}
    watchlist = [(dtrain,'train'),(dtest,'test')]
    XGB_nom = xgb.train(
        params, 
        dtrain, 
        num_rounds,
        watchlist,
        early_stopping_rounds=10) 
        
    #import pickle
    #pickle.dump(XGB_nom, open("XGB_simple_no_nom_f1_9015.pickle.dat", "wb"))

    XGB_predict_label = XGB_nom.predict(dtest)
    plot_AUC(list_label_test,XGB_predict_label)
    for i in [0.4]:
            
        XGB_predict_label_int = predict_label_to_int(XGB_predict_label,threshold=i)
        print('XGB ',i,' : ')
        measure(
                XGB_predict_label_int,list_label_test)
    
    return XGB_predict_label_int

def lr_model(train_data_padding,test_data_padding)->list:
    lr = linear_model.LinearRegression(
        normalize=True
        ,n_jobs=-1)
    lr.fit(train_data_padding, list_label_train)
    result = lr.predict(test_data_padding)

    plot_AUC(list_label_test,result)

    #for i in [0.001, 0.499,0.5,0.501 ]:
    for i in [0.5 ]:
        predict_LinearRegression = predict_label_to_int(
            result,
            threshold= i)
        print('LR ',i,' : ')
        measure(
            predict_LinearRegression,list_label_test)
    return predict_LinearRegression
#%%
def assemble_predicted_and_predict_label(
    list_label,
    list_e_id,
    predict_label,
    mode='eid')-> dict:
    """
    Returns:
        dict: { e_id:[ predicted_label , predict_label ]}
    """    
    if mode =='eid': 
        dict_ori_and_predict_label = {}
        for row in range(len(list_label)):
            e_id = list_e_id[row]
            ori_label = list_label[row]
            dict_ori_and_predict_label[e_id] = [ori_label,predict_label[row]]

        return dict_ori_and_predict_label
    if mode =='TorF':
        dict_ ={
                'TP':[],
                'TN':[],
                'FP':[],
                'FN':[]
            }
        
        for i in range(len(list_e_id)):
            eid = list_e_id[i]
            ori = list_label[i]
            pre = predict_label[i]
            
            if pre==ori: # SUCCESS
                if ori ==1:
                    dict_['TP'].append(eid)
                if ori ==0:
                    dict_['TN'].append(eid)
            else:        # Fail
                if pre ==1:
                    dict_['FP'].append(eid)
                if pre ==0:
                    dict_['FN'].append(eid)
        return dict_


#%% Buliding process
    '''
    dict_data_train = transfor_matrix_process(
        analy_mode=True,
        load_log_from_json = False,
        transfor_matrix = 'complex',
        name = 'train',
        raw_folder_path = 'raw_data_file\\')
    dict_data_test = transfor_matrix_process(
        analy_mode=True,
        transfor_matrix = 'complex',
        load_log_from_json = False,
        name = 'test',
        raw_folder_path = 'raw_data_file\\')
    others_train = choose_data_from_dict('train')
    others_test = choose_data_from_dict('test')
    dict_train = assemble_transfor_matrix_and_others(
        matrix_dict = dict_data_train,
        others_dict= others_train)
    dict_test = assemble_transfor_matrix_and_others(
        matrix_dict = dict_data_test,
        others_dict= others_test)'''
# Running process
#%%
# Complex matrix
dict_train_complex_matrix =json.load(open('Final_Dataset\\transfor_matrix_train.json','r'))
dict_test_complex_matrix = json.load(open('Final_Dataset\\transfor_matrix_test.json','r'))

dict_test_complex_nom = nomilized_matrix(
    dict_test_complex_matrix,
    matrix_type = 'complex')
dict_train_complex_nom = nomilized_matrix(
    dict_train_complex_matrix,
    matrix_type = 'complex')

list_data_train_complex,list_e_id_train,list_label_train = prepare_label(dict_train_complex_nom)
list_data_test_complex,list_e_id_test,list_label_test    = prepare_label(dict_test_complex_nom)

train_data_complex  =np.array( list_data_train_complex)
test_data_complex   =np.array( list_data_test_complex)
Padding = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data_complex_padding = Padding.fit_transform(train_data_complex )
test_data_padding  = Padding.fit_transform(test_data_complex )

#%% complex
xgb_model(train_data_complex,test_data_complex)
lr_model(train_data_complex_padding,test_data_padding)


#%%
# Simple matrix
dict_train_simple_matrix = json.load(open('after_processed_data_file\\dict_data_train_for_analy_fix_birth.json','r'))
dict_test_simple_matrix  = json.load(open('after_processed_data_file\\dict_data_test_for_analy_fix_birth.json','r'))


dict_train_simple_nom = nomilized_matrix(
    dict_train_simple_matrix,
    matrix_type = 'simple',
    ignore=True)
dict_test_simple_nom = nomilized_matrix(
    dict_test_simple_matrix,
    matrix_type = 'simple',
    ignore= True)

list_data_train_simple,list_e_id_train,list_label_train = prepare_label(dict_train_simple_nom)
list_data_test_simple,list_e_id_test,list_label_test    = prepare_label(dict_test_simple_nom)
'''
new_ = []
for i in range(len(list_data_test_simple)):
    new_.append([*list_data_test_simple[i],list_label_train[i]])
import pandas as pd 
pd.DataFrame(new_).to_csv('list_data_train_simple_with_label.csv')
'''
train_data_simple =np.array( list_data_train_simple)
test_data_simple  =np.array( list_data_test_simple)


 
Padding = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data_simple_padding = Padding.fit_transform(train_data_simple)
test_data_simple_padding  = Padding.fit_transform(test_data_simple)

#%%
XGB_RESULT = assemble_predicted_and_predict_label(
    list_e_id=list_e_id_test,
    mode = 'TorF',
    list_label = list_label_test,
    predict_label= xgb_model(train_data_simple,test_data_simple)
    )


LR_RESULT = assemble_predicted_and_predict_label(
    list_e_id=list_e_id_test,
    list_label = list_label_test,
    mode = 'TorF',
    predict_label= lr_model(train_data_simple_padding,test_data_simple_padding)
    )



#%% simple xgb choose 0.4 threshold ,lr choose 0.5
FN = set(XGB_RESULT['FN'])&set(LR_RESULT['FN'])
FP = set(XGB_RESULT['FP'])&set(LR_RESULT['FP'])

#json.dump(XGB_RESULT,open('analy\\XGB_RESULT.json','w'))
#json.dump(LR_RESULT,open('analy\\LR_RESULT.json','w'))
#%% Analy result
import pandas as pd
def choose_data_for_result(model:str,result_dict,dict_data):
    for name,eid_list in result_dict.items():
        data = []
        for eid in eid_list:
            data.append(
                [eid,*dict_data[str(eid)]['features']])
        pd.DataFrame(data,columns = col,).to_csv(model+'_'+name+'.csv')

choose_data_for_result(
    model= 'LR',
    result_dict=LR_RESULT,
    dict_data=dict_test_simple_nom)
#%%

def find_category_dict(ori_dict,mode: str,key:str)-> dict:
    
    if key=='label':
        dict_0 = {}
        dict_1 = {}
        for eid,dict_ in ori_dict.items():

            
            if dict_['label'] ==1:
                dict_1[eid] = dict_
            if dict_['label'] ==0:
                dict_0[eid] = dict_
        
        return dict_0,dict_1

def make_matrix(dict_matrix,mode:str,drop_avg_0_feature=False):
   
    new_dict = {}
    for eid,dict_ in dict_matrix.items():
        
        if mode == 'simple':
            
            m = dict_['log_features'][8:]
            s = sum(m)
            if s!=0:
                m = [ int(1000*i/s) for i in m] # nomolization
           

            matrix =     [
                    [m[0],m[1],m[2],m[3]],
                    [m[4],m[5],m[6],m[7]],
                    [m[8],m[9],m[10],m[11]],
                    [m[12],m[13],m[14],m[15]],
                ]
            new_dict[eid] = matrix
        if mode == 'complex':
            head = 18
            row_len = 22
            m = dict_['features'][head:]
            if drop_avg_0_feature ==True:
                row_len=15
                m= np.array(m)[
                    [
                        0,1,2,3,  # video
                        5,8,9,10, # question
                        11,12,    # comment
                        15,16,17,18,19 # click
                        ]].tolist()

            s = sum(m)

            matrix = np.zeros((row_len,row_len),dtype= 'int')
            i = 0
            if s !=0:
                for row in range(row_len):
                    for col in range(row_len):
                        matrix[row,col] = int(50000*m[i]/s)
                        i+=1
            else:
                for row in range(row_len):
                    for col in range(row_len):
                        matrix[row,col] = int(0)
                        i+=1
            
            new_dict[eid] = matrix
    return new_dict
def compute_avg_matrix(
    dict_matrix,
    mode:str,
    name:str,
    drop_avg_0_feature =True)-> dict:
    
    if mode =='simple':
        dim = 4
        index = ['video','question','comment','click']
    if mode == 'complex':
        dim = 22
        index = [
            'seek_video', 'load_video', 'play_video', 'pause_video', 'stop_video', 
            'problem_get', 'problem_check', 'problem_save', 'reset_problem', 'problem_check_correct', 'problem_check_incorrect', 
            'create_thread', 'create_comment', 'delete_thread', 'delete_comment', 
            'click_info', 'click_courseware', 'close_courseware', 'click_about', 'click_forum', 'close_forum', 'click_progress']

        if drop_avg_0_feature ==True:
            #dim=15
            row = 15
            col = 15
            index = [
                'seek_video', 'load_video', 'play_video', 'pause_video',
                # 'stop_video', 4
                'problem_get', 
                #'problem_check', 'problem_save', 6,7
                'reset_problem', 'problem_check_correct', 'problem_check_incorrect', 
                'create_thread', 'create_comment', 
                #'delete_thread', 'delete_comment', 13 ,14
                'click_info', 'click_courseware', 'close_courseware', 'click_about', 
                'click_forum', 
                #'close_forum', 'click_progress' 20,21
                ]
        
        
    matrix = np.zeros((row,col),dtype = 'int')
   
    for eid,m in dict_matrix.items():
        m_row = 0
        
        for row in range(dim):

            if row in [4,6,7,13,14,20,21]:
                continue
            m_col = 0
            for col in range(dim):

                if col in [4,6,7,13,14,20,21]:
                    continue
                if (m_col==15)or(m_row==15):
                    print(row,col)
                matrix[m_row,m_col]+=m[row][col]
                
                m_col+=1
            m_row+=1

    matrix = matrix/len(dict_matrix)
    
    df_m = pd.DataFrame(matrix,index=index,columns= index)
    sn.heatmap(df_m,linewidths=1,annot=True)
    plt.title(mode.capitalize()+' Transfor Matrix of '+name+' samples')
    return df_m



#%%

dict_test_simple_0,dict_test_simple_1 = find_category_dict(dict_test_simple_matrix,mode ='simple',key ='label')
avg_matrix_0 = compute_avg_matrix(
    make_matrix(dict_matrix = dict_test_simple_0,mode = 'simple')
    ,mode='simple'
    ,name='Negative')
avg_matrix_1 = compute_avg_matrix(
    make_matrix(dict_matrix = dict_test_simple_1,mode = 'simple')
    ,mode='simple'
    ,name='Positive')
#%%
dict_test_complex_matrix = json.load(open('Final_Dataset\\transfor_matrix_test.json','r'))
dict_test_complex_0,dict_test_complex_1 = find_category_dict(dict_test_complex_matrix,mode ='complex',key ='label')
plt.rcParams['figure.figsize'] = (20.0, 20.0)
#%%
m_0= make_matrix(dict_matrix = dict_test_complex_0,mode = 'complex')
   
#%%
avg_matrix_0 = compute_avg_matrix(
   m_0,mode='complex'
    ,name='Negative')
#%%
avg_matrix_1 = compute_avg_matrix(
    make_matrix(dict_matrix = dict_test_complex_1,mode = 'complex')
    ,mode='complex'
    ,name='Positive')


 
# %% SET NAME OF FEATURE
'''
[
    static: 0-7
    
    ,8:gender
    ,9:birth_year
    ,10:edu_degree
    ,11:course_category
    ,12:course_type
    ,13:course_duration
    student_amount,
    course_amount,
    dropout_rate_of_course,
    dropout_rate_of_user
]
'''
#%%
transfor_node = [
    'seek_video', 'load_video', 'play_video', 'pause_video', 'stop_video', 
    'problem_get', 'problem_check', 'problem_save', 'reset_problem', 'problem_check_correct', 'problem_check_incorrect', 
    'create_thread', 'create_comment', 'delete_thread', 'delete_comment', 
    'click_info', 'click_courseware', 'close_courseware', 'click_about', 'click_forum', 'close_forum', 'click_progress']
feature_dict    = {
     0:'L_mean',1:'L_var',2:'L_skew',3:'L_kurtosis',
     4:'S_mean',5:'S_var',6:'S_skew',7:'S_kurtosis'
    ,8:'gender'
    ,9:'birth_year'
    ,10:'edu_degree'
    ,11:'course_category'
    ,12:'course_type'
    ,13:'course_duration'
    ,14:'student_amount',
    15:'course_amount',
    16:'dropout_rate_of_course',
    17:'dropout_rate_of_user'}
i = 18
for start in transfor_node:
    for end in transfor_node:
        scene = str(start)+'_TO_'+str(end)
        feature_dict[ i ] = scene
        i+=1

# %%
import numpy as np
import pandas as pd
useful = []
for i in WEIGHT:
    name = feature_dict[WEIGHT.index(i)]
    if np.abs(i)>0.005:
        useful.append([name,i])
useful_pd = pd.DataFrame(useful)

# %%

# %%

col = [
'gender', 'birth_year', 'edu_degree', 
'course_category', 'course_type', 'course_duration', 
'student_amount', 'course_amount', 'dropout_rate_of_course', 'dropout_rate_of_user',
'L_mean', 'L_var', 'L_skew', 'L_kurtosis', 
'S_mean', 'S_var', 'S_skew', 'S_kurtosis',
'11','12','13','14',
'21','22','23','24',
'31','32','33','34',
'41','42','43','44','label'
 ]
# %%



from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nGraphShown = 15
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    plotSize = 30
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    #if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
    #    columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = None # specify 'None' if want to read whole file
# count_vect.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/count_vect.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'count_vect.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
        