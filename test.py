import pandas as cudf
from pandas import DataFrame
from numpy import ndarray
from datetime import datetime
from typing import List, Dict 
import os
import json
def return_file(path: str)-> Dict[str,str]:
    path_dict = {}    
    for dirname, _, filenames in os.walk(path):    
        
        for filename in filenames:            
           
            path = os.path.join(dirname, filename)
            path_dict[filename] = path
            #print(path)
            #yield path       
    return path_dict          
def load_data(path_dict: Dict[str,str],base: str)-> Dict[str,DataFrame]:
    print('load run' )    
    raw_data = {}
    if base == 'cudf':
        print('read csv by cudf')
        for file in path_dict:
            raw_data[file[:-4]] = cudf.read_csv(path_dict[file])
               # dtype = columns_type_dict[file], 
            print(
                file[:-4],':',"\n",
                "Rows: {:,}".format(len(raw_data[file[:-4]])), "\n" +
                "Columns: {}".format(len(raw_data[file[:-4]].columns)),'\n')
    print('load finish by ',base)
    return raw_data
def find_null_set_null(raw_dict:Dict[str,DataFrame],base: str,defult: int)->DataFrame:
    print('find_null run')
    for i in raw_dict:
        print(i,' : \n')
        raw_data = raw_dict[i]
        for i in raw_data:    
            print(i,' :')
            total = len(raw_data[i])
            for column in raw_data[i].columns:
                if raw_data[i][column].isna().sum() != 0:
                    
                    print("{} has: {:,} ,{:.2f}% missing values.".format(column, raw_data[i][column].isna().sum(), 
                                                                        (raw_data[i][column].isna().sum()/total)*100))
                    raw_data[i][column] = raw_data[i][column].fillna(defult) # set null = -1
                else : print(column,' ','does not find null')
            print('\n')
        raw_dict[i] = raw_data
    return raw_dict
def print_nunique_in_each_col(df:DataFrame)->None:
    for col in train_log.columns:
        print( 'in ',col,' : ',
            train_log[col].unique(),'\n',
            train_log[col].nunique(),' unique values'
            )
def preprocess(log:DataFrame)-> Dict[str,Dict[int,DataFrame]]:
    enroll_dict  = {}
    course_dict  = {}
    student_dict = {}

    for i,v in log.iterrows() :
        # init
        if v['enroll_id'] not in enroll_dict:
            enroll_dict[v['enroll_id']] = cudf.DataFrame()
        if v['course_id'] not in course_dict:
            course_dict[v['course_id']] = cudf.DataFrame()
        if v['username'] not in student_dict:
            student_dict[v['username']] = cudf.DataFrame()
        # init end
        enroll_dict[v['enroll_id']]  = enroll_dict[v['enroll_id']].append(
            v.drop(labels =['enroll_id','username','course_id']))
        # enroll_dict[i] including--> 'action''time''object'

        if v.username not in course_dict[v['course_id']].values:
            course_dict[v['course_id']]  = course_dict[v['course_id']].append([v.username])
        if v.course_id not in student_dict[v['username']].values:
            student_dict[v['username']]  = student_dict[v['username']].append([v.course_id])

    # sort
    for i in enroll_dict:
        enroll_dict[i] = enroll_dict[i].sort_values(by = ['time'])
    
    def filter(enroll_dict)-> Dict[int,DataFrame]:
        for i in enroll_dict:
            if len(enroll_dict[i])<5 :
                del enroll_dict[i]
        return enroll_dict
    enroll_dict = filter(enroll_dict)


    processed_data= {
        'enroll':enroll_dict,
        'course':course_dict,
        'student':student_dict }

    
    return processed_data
#def time_convert(log:DataFrame)->log:DataFrame:
#    pass
def tokenlization(s):
    actions = set()
    # actons.update()

    return s

def to_vec(dict_enroll : Dict[int,DataFrame])->DataFrame:
    def find_gap(i : int)->Dict[str,int]:
        t = {'start':0, 'end':0}
        return t
    def incert_time(df : DataFrame, start : int, end : int )->ndarray:
        log = DataFrame.values
        return log
    def   zip_log(log : ndarray, defult_len : int)->ndarray:
        return log_zip
    for i in dict_enroll :
        start,end = find_gap(i)
        log = incert_time(dict_enroll[i],start,end)
        log_zip = zip_log(log,defult_len)
        # dataframe to time_series 
        #   package by dict {[time,action],[time,action].....}
    return  # action_series_dict

def seriesNN(action_series):
    network = fit(action_series)
    return network_para

'''
path = 'D:\\zyh\\_workspace\\dropout_prediction\\test'
path_dict = return_file(path)
df =  find_null_set_null(  load_data(path_dict, base = 'cudf')  ,base ='cudf',defult= -1)
train_log = df['train_log']
user_id = df['user_info'].index.unique()

'''

a = np.ndarray(
    [ #1
''' e1'''
        [ #2
        [e1,a,s,o] , # 3
        [e1,a,s,o]
        ],
''' e2'''
        [
        [e2,a,s,o],
        [e2,a,s,o]
        ]
    ])