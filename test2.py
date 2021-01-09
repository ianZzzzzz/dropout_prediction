from typing import List, Dict
from pandas import DataFrame
from numpy import ndarray
#import cudf
import numpy as np # linear algebra
import pandas as cudf # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings

#warnings.simplefilter(action='ignore', category=FutureWarning)
def unique_count(df):
    print(df.dataframeName,'unique_count running!')
    for i in df.columns:
        object_ = df[i]
        print(i,' count:',object_.nunique())
def data_utilization(ori,processed):
    print('origin len : ',           len(ori),
          '\n','after processed len : ',len(processed),
          '\n','data utilization : ',(len(processed)/len(ori))*100,'%')
def filter_3sigma(df_,key):
    df= df_[key]
    head = df.mean()+3*df.std()
    tail = df.mean()-3*df.std()
    df_ = df_[(df_[key]<=head) & (df_[key]>=tail)]
    return df_
def cut_nan(df,label):
    df = df[~df[label].isna()]
    return df
def percent_filter(df,key,up,down):

    series = df[key].sort_values()
    up = series.quantile(up)
    down = series.quantile(down)
    df = df[(df[key]<=up.squeeze()) & (df[key]>=down.squeeze())]
    return df
def preprocess(log:DataFrame)-> Dict[str,Dict[int,DataFrame]]:
    enroll_dict  = {}
    course_dict  = {}
    student_dict = {}
    k=0
    print('pre running!')
    for i,v in log.iterrows() :
        # init
        k+=1
        if k%1000==0:
            print('iterate ',k)
            
        if v['enroll_id'] not in enroll_dict:
            enroll_dict[v['enroll_id']] = cudf.DataFrame()
        if v['course_id'] not in course_dict:
            course_dict[v['course_id']] = cudf.DataFrame()
        if v['username'] not in student_dict:
            student_dict[v['username']] = cudf.DataFrame()
        # init end
        frame = cudf.DataFrame([v.drop(labels =['enroll_id','username','course_id'])])
        enroll_dict[v['enroll_id']]  = enroll_dict[v['enroll_id']].append(frame)
        # enroll_dict[i] including--> 'action''time''object'

        if v.username not in course_dict[v['course_id']].values:
            course_dict[v['course_id']]  = course_dict[v['course_id']].append([v.username])
      #  if v.course_id not in student_dict[v['username']].values:
       #     student_dict[v['username']]  = student_dict[v['username']].append([v.course_id])

    # sort
    '''  for i in enroll_dict:
        enroll_dict[i] = enroll_dict[i].sort_values(by = ['time'])'''
    
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

nRowsRead = None # specify 'None' if want to read whole file
# test_log.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
ori_test = cudf.read_csv('D:\\zyh\\_workspace\\dropout_prediction\\prediction_log\\test_log.csv', delimiter=',', nrows = nRowsRead,index_col = None)
ori_test.dataframeName = 'test_log.csv'
nRow, nCol = ori_test.shape
print(f'There are {nRow} rows and {nCol} columns')
#unique_count(ori_test)
print(1)
test_log = preprocess(ori_test)
print(2)