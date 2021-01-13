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

        __time = log_np[:,1].astype('int')

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
                # 为了省内存 将不同的action用数字代替 都是符号 不影响数据特征 
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
                
                if _action in replace_dict:
                    _action = replace_dict[_action]
                else:
                    _action = b'0'
                list_action[row_num] = _action

            except:
                print('ERROR log time [_time] :',_time)
                
                print('list_time :',list_time,'list_action :',list_action)
                
        
        rebulid = np.concatenate( (list_action, list_time), axis = 1)
        
        '''出于保留 ‘用户主要的操作分布在开课时间的哪一部分’ 这一特征 
            的目的，将时间转换部分分为两部分写，后期如需重建此特征以上的代码可以不动'''

        action_series = time_map(rebulid)
        new_dict[e_id] =  action_series 
        
        
    return new_dict


log_np_convert = convert(log_dict,drop_zero = True)

key =0
for e_id ,v in log_np_convert.items():
    print('e:',e_id,'v :',v)
    break

