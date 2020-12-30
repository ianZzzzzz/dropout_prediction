from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
#import cudf
import json
import pandas as cudf
import numpy as np
''' @ log process
    .csv--> load_df(raw_log_path) --df--> df_to_dict(df) 
        --dict--> to_json -->.txt --> load_dict_list(dict_path)

    @ course info process
    .csv--> course_info_process(info_path) --> ndarray

'''
raw_log_path = 'D:\\zyh\\_workspace\\dropout_prediction\\prediction_log\\test_log.csv'
def load_df(path)->DataFrame:
    ''' .csv --cudf or pandas--> df
        df-->count unique values in each columns
        return df
    '''
    def unique_count(df):
        print(df.dataframeName,'unique_count running!')
        for i in df.columns:
            object_ = df[i]
            print(i,' count:',object_.nunique())
    
    nRowsRead = None 
    ori = cudf.read_csv(path, delimiter=',', nrows = nRowsRead,index_col = None)
    nRow, nCol = ori.shape
    print(f'There are {nRow} rows and {nCol} columns')
    unique_count(ori)
    return ori

def df_to_dict(pd_log:DataFrame,c_info: ndarray)->Dict[int,ndarray]:
    ''' df columns: enroll_id,  # useful
                    course_id,
                    username,
                    action,     # useful
                    session_id,
                    object,
                    time        # useful
        
        df --.values--> array
        array --choose useful columns--> array
        array --> dict[ enroll_id   :   array.tolist()  ]
        @return dict
    '''
    def time_padd(log_for_one:ndarray,c_dict: ndarray)->ndarray:
        ''' 
        
        '''
        for row in range(len(log_for_one)):
            log_for_one[row,2]= np.datetime64(log_for_one[row,2])
            start = log_for_one[row,2]
            course_id = log_for_one[row,3]
            end = c_dict[course_id][0]
            log_for_one[row,2] = int((start - end).item().total_seconds())
        # 将log里时间字符串转换成秒数
        def gen_empty_array():   
            empty = np.zeros(c_dict[course_id][1],np.int32)

    pass
    np_log = pd_log.values
    np_log = np_log[:,[0,4,6,2]] # enroll_id, action, time， course_id
    enroll_id_set = set(np_log[:,0]) # find unique values
    enroll_log_dict = {}
    c = 0
    import time
    total_silce = 0
    total_dict = 0
    loop_start = time.clock()
    for i in enroll_id_set:
        

        mask = np_log[:,0]==i
        #slice_s = time.clock() #
        log_for_one = np_log[mask][:,[1,2,3]]

        #

        #slice_e = time.clock() #
        #slice_g = slice_e - slice_s##
        #enroll_log_dict[i] = log_for_one.tolist() # 写入json才做
        #dict_e = time.clock() #
        #dict_g = dict_e -slice_e##
        #total_silce = total_silce + slice_g #
        #totao_dict = total_dict +dict_g #
        c+=1
        if c%1000 == 0 :
            print('loop count : ',c,
        #        'slice time : ',total_silce,
        #        'dict time : ',total_dict
            )
    loop_end = time.clock()
    loop_g = loop_end - loop_start
    print('loop time : ',loop_g)
    return enroll_log_dict

def to_json(path,data: Dict[int,ndarray]):
    '''dict[    enroll_id : array.tolist()]-->json_txt
        # json不支持ndarray
        # 用json导出 array 要先 .tolist() 读取的时候直接np.array()
    '''
    import json
    json.dump(data,open(path,'w'))

def load_dict_list(path)-> Dict[int,ndarray]:
    '''.txt-->dict[ enroll_id , list ]
        list --np.array(list)--> array
        
    '''
    import time
    import json
    l_s = time.clock()
    dict_log = json.load(open(path))
    s = time.clock()
    for i in dict_log:
        dict_log[i] = np.array(dict_log[i])
    e = time.clock()
    print('dict len : ',len(dict_log),'load time : ',s-l_s,'to array time : ',e-s)
    return dict_log

def cut_nan(df,label):
    df = df[~df[label].isna()]
    return df
dict_path = 'D:\\zyh\\data\\enroll_dict.txt'
np_dict_log = load_dict_list(dict_path)

def course_info_process(info_path)->ndarray:
    '''.csv columns:id,          # also course_id but ues in tracking log dataset
                    course_id,
                    start,
                    end,
                    course_type,
                    category

        .csv --pd or cudf-->dataframe
        df --cut_nan--> df
        df --.values--> ndarray
        @ return ndarray
                    '''
    course_info = cudf.read_csv(info_path)
    course_info = cut_nan(course_info,'end').values
    for row in range(len(course_info)):
        course_info[row,2]= np.datetime64(course_info[row,2])
        course_info[row,3]= np.datetime64(course_info[row,3])

        course_info[row,3]= int((course_info[row,3] - course_info[row,2]).item().total_seconds())
    return course_info
info_path = 'D:\\zyh\\_workspace\\dropout_prediction\\test\\course_info.csv'
c_info = course_info_process(info_path)

def dict_info(c_info: ndarray)-> Dict[int,ndarray]:
    c_info = c_info[:,[1,2,3]]
    c_dict = {}
    for row in range(len(c_info)):
        id =  c_info[row,0]
        c_dict[id] = c_info[row,[1,2]]
    return c_dict

#c_dict = dict_info(c_info)
