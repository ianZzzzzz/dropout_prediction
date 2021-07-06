#%%
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame

def Major_data_process(name:str,raw_folder_path:str):
    TEST_OR_NOT = False
    print_batch = int(1000000)
    chunk_size  = int(10000) # enable only when TEST_OR_NOT = True
    from scipy import stats
    import pandas as pd
    import numpy as np
    import json

    def extract_feature(name:str,dict_log,return_node = 'dict')-> dict:
        """[info feature : dropout rate of users and courses]

        Args:
            name (str): [description]
            dict_log ([type]): [description]
        Return:
            [
              
           
        0_L_mean,# long interval
        1_L_var,
        2_L_skew,
        3_L_kurtosis, 
        4_S_mean,# short interval
        5_S_var,
        6_S_skew,
        7_S_kurtosis 

        # transfor matrix 4*4
        8_11 ,9_12 ,10_13,11_14,
        12_21,13_22,14_23,15_24,
        16_31,17_32,18_33,19_34,
        20_41,21_42,22_43,23_44,

        24_gender
        ,25_birth_year
        ,26_edu_degree
        ,27_course_category
        ,28_course_type
        ,29_course_duration,
        30_student_amount,
        31_course_amount,
        32_dropout_rate_of_course,
        33_dropout_rate_of_user
        ]
        important :
            184 long interval mean
            176 dropout_rate_of_user
            157 dropout_rate_of_course
            138 long interval var
            127 2->2 action
            120 4->4 action
            102 short interval mean
            100 course_amount
            99  1->1 action

        """        
        
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
            THERSHOD_long_interval = int(60*5)
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
                            np.nan,np.nan ,np.nan ,np.nan ]
                        
                    
                    return static_list
            print("\n    extract_feature_on_LogData running : \n")
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
                        
                        if time_interval >THERSHOD_long_interval: # 
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
                if (i%5000)==0:
                    print('already processed  ',i,' enrollment id')

                # time_interval_dict[int(e_id)] = interval_list
                long_static.extend(short_static)
                static_list = long_static
                # static_interval_dict[int(e_id)] = static_list
                # enroll_scene_dict[int(e_id)] = scene_list
                static_list.extend(scene_list)
                static_and_scene_list = static_list
                static_and_scene_dict[int(e_id)] = static_and_scene_list
                # break
            
            print('      extract_feature_on_LogData finish')
            print('      Success extract interval static values and actions transfer matrix.')
            return static_and_scene_dict #static_interval_dict,enroll_scene_dict
        
        def extract_feature_on_InfomationData(
            mode: str,
            threshold_course_amount = int(3),
            threshold_student_amount = int(3))-> dict:
            """[caculate_droupout_rate]

            Returns:
                [type]: [
                    gender
                    ,birth_year
                    ,edu_degree
                    ,course_category
                    ,course_type
                    ,course_duration,
                     student_amount,
                     course_amount,
                     dropout_rate_of_course,
                     dropout_rate_of_user
                ]
            """            
            # prepare for hot data
            # unsuitable for cold start
            # need thershod to choose enable or not
            # 训练结果可以关联到课程分类上
            def load(
                log_path: str,
                return_mode: str,
                encoding_='utf-8',
                read_mode = 'pandas',
                columns=None,
                test=TEST_OR_NOT)-> ndarray or DataFrame:
                '''读取csv文件 返回numpy数组'''
            
                if read_mode == 'pandas' :
                    import pandas as pd
                    if test ==True: # only read 10000rows 
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
                        print('      Start loading ',log_path)
                        log = pd.read_csv(
                            log_path
                            ,encoding=encoding_
                            ,names=columns)
                        print('        Total length :',len(log),'rows.')
                    
                if return_mode == 'df'      :return log
                if return_mode == 'ndarray' :return log.values
                if return_mode == 'list'    :return log.values.tolist()
                    
            def assemble_info_data(name:str)-> dict:
                """[concat user info and course info ,index by enroll id]

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
                            print('          Start loading ',log_path)
                            log = pd.read_csv(
                                log_path
                                ,encoding=encoding_
                                ,names=columns
                                ,low_memory=False)

                            
                            print('            Total length :',len(log),'rows')
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
                        'nan':0
                        , 'male' :1
                        , 'female' :2}    
                    education_degree_replace_dict = {
                        'nan': 0
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
                        
                        '''if start_time is np.:
                            start_time = nan                    else:
                            start_time = np.datetime64(start_time)
                        if end_time is np.:
                            end_time = nan                    else:
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
                    print('        Success concat and filter out ',name,' infomation in course_info and user_info . ')
                            
                    if return_mode == 'dict':
                        return dict_enroll_info
                    else: 
                        if  return_mode == 'list':
                            return    list_enroll_info

                print('      assemble_info_data running : ')
                c_info_path = 'raw_data_file\\course_info.csv'
                u_info_path = 'raw_data_file\\user_info.csv'
                u_info_col = ['user_id','gender','education_degree','birth_year']
                c_info_col = ['id','course_id','start_time','end_time','course_type','course_category']
                
                # load original info file
                print('        Loading user info:')
                U_INFO_NP = load(
                    log_path =u_info_path,
                    read_mode ='pandas',
                    return_mode = 'values',
                    encoding_ = 'utf-8',
                    columns =u_info_col)
                
                print('        Loading course info:')
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

                print('      assemble_info_data finish.')
                return info_dict_eID

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
            
            
            
            print('    extract_feature_on_InfomationData running : ')
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
            # {u_id:[course_amount,drop_rate]}
            users_courseAmount_and_drop_rate    = json.load(
                open('extracted_features\\users_courseAmount_and_drop_rate.json','r'))
            
            # {c_id:[student_amount,drop_rate]}
            courses_studentAmount_and_drop_rate = json.load(
                open('extracted_features\\courses_studentAmount_and_drop_rate.json','r'))
            
            raw_course_and_user_info_dict         = assemble_info_data(name = mode)
            # Assemble
            print('      Start choose infomation features for ',mode,' data.')
            e_id_list = list(enroll_find_course.keys())
            info_feature_dict = {}

            for e_id in e_id_list:

                e_id = str(e_id) # load from json need str type key
                c_id = enroll_find_course[e_id]
                u_id = str(enroll_find_user[e_id])


                try:
                    student_amount         = courses_studentAmount_and_drop_rate[c_id][0],
                    student_amount = student_amount[0]
                except:
                    student_amount = np.nan
                try:
                    course_amount          = users_courseAmount_and_drop_rate[u_id][0],
                    course_amount = course_amount[0]
                except:
                    course_amount = np.nan
                try:
                    dropout_rate_of_course = courses_studentAmount_and_drop_rate[c_id][1],
                    dropout_rate_of_course = dropout_rate_of_course[0]
                except:
                    dropout_rate_of_course = np.nan
                try:
                    dropout_rate_of_user   = users_courseAmount_and_drop_rate[u_id][1]
                    #dropout_rate_of_user   = dropout_rate_of_user[0]
                except:
                    dropout_rate_of_user   = np.nan


                e_id = int(e_id)
                info_feature_dict[e_id] = [
                     *raw_course_and_user_info_dict[e_id],
                     student_amount,
                     course_amount,
                     dropout_rate_of_course,
                     dropout_rate_of_user
                    ]
            print('    Extract Infomation features finish.')
            return info_feature_dict

        print('  Extract_feature running : ')
        #print('    Extracting ',name,'features :')


        if name == 'train':
            
            # info_dict         = assemble_info_data(name = 'train')
            log_feature_dict  = extract_feature_on_LogData(dict_log)
            info_feature_dict = extract_feature_on_InfomationData(mode = 'train')
            # assemble
            
        if name == 'test':
            
            # info_dict         = assemble_info_data(name = 'test')
            log_feature_dict  = extract_feature_on_LogData(dict_log)
            info_feature_dict = extract_feature_on_InfomationData(mode = 'test')
            
        print('  Extract_feature finish. ')
        if return_node == 'list':   
            Features = []
            row = 0  
            for e_id in log_feature_dict.keys():
                Features[row] = [
                   # *info_dict[e_id],
                    *log_feature_dict[e_id],
                    *info_feature_dict[e_id] ]

                row+=1
            print('\n  ALL features are ready to use.\n')
            return Features
        if return_node == 'dict':   
            Features = {}
            for e_id in log_feature_dict.keys():
                e_id = int(e_id)
                Features[e_id] = [
                  #  *info_dict[e_id],
                    *log_feature_dict[e_id],
                    *info_feature_dict[e_id] ]
            print('  ALL features are ready to use.')
            return Features

    def load_label(mode:str,id_list: list)->list:
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
        
        print('  load_label running : ')
        np_label = load(
        
            log_path = raw_folder_path+mode+'_truth.csv')
        
        dict_label = list_to_dict(list_ = np_label.tolist())

        print('  load_label finish.\n')
        return list(dict_label.values())
    
    print('\n',name.capitalize()+'ing Mode Enable .')
    print('Major Data Process is running : \n') 
    export_path = 'after_processed_data_file\\'+name+'\\dict_'+name+'_log_ordered.json'
    dict_log = json.load(open(export_path,'r'))    
   
    dict_data = extract_feature(
        name = name,
        dict_log = dict_log
        )
    dict_log = None # release ram

        
    list_data  = list(dict_data.values())
    list_label = load_label(
        mode = name,
        id_list=list(dict_data.keys()))
    
    
    json.dump(
        list_data,
        open('Final_Dataset_fix\\'+name+'_data.json','w'))
    json.dump(
        list_label,
        open('Final_Dataset_fix\\'+name+'_label.json','w'))
    print('Major Data Process is finish.')     
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
                'courseware-video','courseware-answer','courseware-comment','courseware-courseware',
                'course_amount','dropout rate of course',
                'student_amount',' dropout rate of user']
            )

        df_label = pd.DataFrame(
            data=label,
            columns=['drop_or_not']
        )
        return df_data,df_label

    # return to_df(list_data,list_label) 
    
    return np.array(list_data),np.array(list_label)

train_data,train_label = Major_data_process(
    name = 'train',
    raw_folder_path = 'raw_data_file\\')
test_data,test_label = Major_data_process(
    name = 'test',
    raw_folder_path = 'raw_data_file\\')


#%%
class Model:
    def __init__(self, mode, data, label):
        self.mode = mode
        self.data = data
        self.label = label
        self.model = None

    def measure(self,mode:str):
        pass
    def train(self ):
        def show_info(self):
            pass
        def preprocess(data):
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            std=StandardScaler()
            mm = MinMaxScaler()
            si = SimpleImputer(np.nan,'mean')
            pipeline=make_pipeline(
                si,
                std,
                mm)
            data = pipeline.fit_transform(data)
            return data
        from xgboost import XGBClassifier # sklearn style api   
        params = {}
        XGB = XGBClassifier(
            learning_rate= 0.05,
            max_depth=10
        )
    
  

        self.model = XGB.fit(
            X=preprocess(self.data),
            y=self.label)

        
    def predict(self,test_data,test_label):
        from sklearn import metrics
        predict_label = self.model.predict(test_data)
       
        print(
            "Accuracy : %.4g" % metrics.accuracy_score(test_label,predict_label)
        )
        
    def save_model(self,path: str):
        pass
    def load_model(self,path: str):
        pass

    def __str__(self):
        return ' Name : {}, Age : {}, Gender : {} '.format(
            self.mode,self.data,self.label)

xgb_model = Model(
    mode = 'xgboost',
    data = df_train_data.values,
    label= df_train_label.values)

xgb_model.train()

xgb_model.predict(
    test_data  = df_test_data.as_matrix(),
    test_label = df_test_label.as_matrix())


