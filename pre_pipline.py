'''程序功能：

        读取日志文件
        根据注册号归类数据行
        以字典方式存储各个注册号下的日志
        在各注册号下的日志中
        取时间值最小与最大的行作为时间起始与重点 得出时间序列长度L 单位秒
        建立长度L 的零数组 将数据行根据时间列的值填充进数组中
        将数据行action列的数据按照操作名替换表替换为数字 以节省内存
        
        '''
'''
不足： 
        数据稀疏
改进：  
        权衡了训练速度与特征完整性 决定目前先去除序列中的零
        所以不保留action间的间隔 只保留acton间的次序 
            （主观默认间隔长短不如间隔次序重要 
        
        还未想到该如何量化的改进：action中观看视频的时间长短我觉得是很重要的特征。
        
        

    上周进展 ：
      读取csv数据对其归类存字典 转换成训练数据 
      特征包含：
        1 不同action 的组合模式与先后模式
        2 action之间的间隔时间
        3 用户主要的操作分布在开课时间的哪一部分

     考虑后决定暂不使用特征3 因为使用完整的序列会使得训练样本太稀疏 导致训练缓慢

     编码：
        怎么选择合适的编码将22种符号特征编码成向量 
        要求近似的行为编码的余弦相似度等指标也接近

        onehot显然不合适 选择词嵌入方式训练 
        AAAI19论文的代码中将行为分成了四类，

'''


# preprocess of "log file to time series"
import pandas as pd
# import cudf as pd # nvidia GPU only # !pip install cudf 
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import numpy as np

TEST_OR_NOT = True
print_batch = int(1000000)
chunk_size = int(1000000)

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

@_t
def plot_histogram(log_list:list):
    '''
        描绘列表中序列长度分布的直方图 
    '''
    from matplotlib import pyplot as plt 
    import numpy as np
    plt.hist([len(s) for s in log_list], bins = 100) # 横坐标精度

    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

    ''' 细节版
        len_array = np.zeros( len(log_list)+1,dtype = np.uint32)
        for i in range(len(log_list)):
            length =  len(log_list[i])
            len_array[i] = length

        series = len_array

        min_ = int(np.min(series))
        max_ = int(np.max(series))
        gap =  max_ - min_

        bins_ = [
            min_
            ,(min_+int(0.10*gap))
            ,(min_+int(0.20*gap))
            ,(min_+int(0.30*gap))
            ,(min_+int(0.40*gap))
            ,(min_+int(0.50*gap))
            ,(min_+int(0.60*gap))
            ,(min_+int(0.70*gap))
            ,(min_+int(0.80*gap))
            ,(min_+int(0.90*gap))
            ,max_
            ]

        plt.hist( series, bins =  bins_)
        plt.show() '''
    


@_t
def word_counter(log_list:list)-> Dict[str,dict]:
    '''
        计算单词频次
    '''
    from matplotlib import pyplot as plt 
    import numpy as np

    action_type_counter = { '1':0,  '2':0, '3':0, '4':0 }
    action_counter = {
        11:0,12:0,13:0,14:0,15:0       # video
        ,21:0,22:0,23:0,24:0,25:0,26:0  # problem
        ,31:0,32:0,33:0,34:0            # common
        ,41:0,42:0,43:0,44:0,45:0,46:0  # click
        }
    for i in range(len(log_list)):

        series_ = log_list[i]
        
        for action_ in series_:

            action_type = str(action_)[0]
            action_type_counter[action_type] +=1
            action_counter[action_] +=1
    
    re = {'type':action_type_counter ,'action':action_counter}
    return re

@_t
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

@_t
def to_dict_2(
    log: ndarray
    ,test=TEST_OR_NOT
    )-> Dict[str,ndarray]: 
    '''优化版 时间复杂度低'''
    print('find ',len(log),' row logs')
    i = 0
    log_dict = {}
    for row in log:
        area = row[0]
        data = row[[1,2,3]]
        #print('area: ',area,' data: ',data)
        
        # if log_dict[]里没数据：初始化=[]
        try:
            log_dict[area].append(data)
        except:
            log_dict[area] = []
            log_dict[area].append(data)
            #print(log_dict[area])
        i+=1
        if (i%print_batch)==0:print('already dict : ',i,'row logs')
        if (test == True) and (i ==print_batch):return log_dict
    return log_dict

@_t
def convert(
    log: dict 
    ,drop_zero: bool
    ,testing=TEST_OR_NOT
    )->dict: 
    print('dict total len :',len(log))
    print(' convert running!')
    
    import numpy as np
    
    def find_start_end(c_id:str)->Dict[int,datetime64]:

        ''' 根据course_id 查询课程的总耗时秒数 以及开始时间并返回
            函数调用了全局变量C_INFO_NP必须在课程信息被加载后才能运行'''
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
        '''action_series改成无零的action有序表 '''
        def to_int(x):
            return int(x)
        md = map(to_int,log_np[:,1]) 
        __time = list(md) # time list

        __head = np.min(__time)
        __tail = np.max(__time)
        __length = __tail - __head +1
        action_series = np.zeros((__length,1),dtype=np.uint8)

        for row in log_np:
            __t = int(row[1]) # time now
            __location = __t - __head
            action_series[__location] = row[0]
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
        # 遍历字典 
        # type(v)==list
        v = np.array(v)
        c_id = v[0,2] 
        # 上一函数为了方便 保留了每一行的course_id 此处只需要一个值
        _log       = v[:,[0,1]]
        time_col   = _log[:,1]
        action_col = _log[:,0] 
        
        time_info   = find_start_end(c_id)
        time_head   = time_info['head']
        time_length = time_info['length']

        list_time   = np.zeros((len(_log),1),dtype = np.uint32)
        list_action = np.zeros((len(_log),1),dtype = 'a32') # a32 存32个英文字符
        

        for row_num in range(len(_log)):
            ''' 为了对action做进一步编码 保留了action接口 '''

            _row = _log[row_num,:]
            _time = _row[1]
            _action = _row[0]
            try:
                _time = np.datetime64(_time)
                _time =  int(
                    ( _time - time_head ).item().total_seconds() )
                list_time[row_num] = _time
                # 为了省内存 将不同的action用字母表代替 都是符号 不影响数据特征 
                replace_dict = {
                    # video
                    'seek_video': 11
                    ,'play_video':12
                    ,'pause_video':13
                    ,'stop_video':14
                    ,'load_video':15
                    # problem
                    ,'problem_get':21
                    ,'problem_check':22
                    ,'problem_save':23
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
                    ,'click_about':43
                    ,'click_forum':44
                    ,'click_progress':45
                    ,'close_courseware':46}
        
                _action = replace_dict[_action]
                list_action[row_num] = _action
            except:print('ERROR log time [_time] :',_time)
        rebulid = np.concatenate( (list_action, list_time), axis = 1)
        

        '''出于保留 ‘用户主要的操作分布在开课时间的哪一部分’ 这一特征 
            的目的，将时间转换部分分为两部分写，后期如需重建此特征以上的代码可以不动'''

        action_series = time_map(rebulid)
        new_dict[e_id] =  action_series 
        if (i ==1000)and(testing ==True): 
            return new_dict
        
    return new_dict

'''此分类引用自AAAI19那篇论文的代码
    video_action = ['seek_video','play_video','pause_video','stop_video','load_video']
    problem_action = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
    forum_action = ['create_thread','create_comment','delete_thread','delete_comment']
    click_action = ['click_info','click_courseware','click_about','click_forum','click_progress']
    close_action = ['close_courseware']
'''
@_t
def cut_nan(
    log: ndarray,
    key_col:list
    )->ndarray:
    '''删除给定列中存在空值的日志行'''
    import numpy as np

    pandas_ = type(pd.DataFrame([]))
    numpy_  = type(pd.DataFrame([]).values)
        # type(type(pd.DataFrame([]))) == <class 'type'> not a str :)

    if type(log)== pandas_ :
        log = log.values
    if type(log)== numpy_:
        pass

    # nan row filter
    for i in key_col:
        mask = np.isnan(log[:,i]) # bug
        log = log[~mask]
    
    return log
@_t
def to_json(path,dict_log: Dict[int,ndarray]):
    '''dict[    enroll_id : array.tolist()]-->json_txt
        # json不支持ndarray
        # 用json导出 array 要先 .tolist() 读取的时候直接np.array()
    '''
    for k,v in dict_log.items():
        dict_log[k] = v.tolist()

        import json
        json.dump(dict_log,open(path,'w'))

@_t
def dict_to_array(dict_log:dict)->list:

    ''' 函数功能:   归类后的数据被存储为dict格式 需要将其转换为list以制作数据集
                  创建空表，将每次读取到的序列追加进表内
        need:   numpy
        note:   用list append执行很快 np.concatenate慢十倍以上'''
    i = 0
    print_key = 10000
    
    for k,v in dict_log.items():
        
        data = np.array(v)
        
        
        try:
            dataset.append(data)
        except:
            dataset = []
            dataset.append(data)
        
        i+=1
        if (i%print_key)==0:
            print('already to array ',i,' areas.')

    return dataset


log_path = 'D:\\zyh\\data\\prediction_data\\prediction_log\\test_log.csv'
log_col = ['enroll_id','username','course_id','session_id','action','object','time']
c_info_path = 'course_info.csv'
c_info_col = ['id','course_id','start','end','course_type','category']

log_np = load(
    log_path =log_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =log_col)
log_dict = to_dict_2( log_np[1:,[0,4,6,2]]) # e_id action time c_id
C_INFO_NP = load(
    log_path =c_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =c_info_col
    )

log_np_convert = convert(log_dict,drop_zero = True)
log_list = dict_to_array(log_np_convert)

plot_histogram(log_list) 


# 得到描述序列长度分布的直方图 以确定截断和填充的长度


# after_convert_path = 'D:\\zyh\\data\\prediction_data\\after_convert_dict\\test_1.json'
# BUG ON  to_json(after_convert_path,log_np_convert)
# TypeError: Object of type ndarray is not JSON serializable



