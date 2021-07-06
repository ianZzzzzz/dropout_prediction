
raw_log_column_name = [
     'enroll_id'
    ,'username'
    ,'course_id'
    ,'session_id'
    ,'action'
    ,'object'
    ,'time']
# 原始数据  
raw_log_from_csv = [
     [775,1520977,'course-v1:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','load_video','3dac5590435e43b3a65a9ae7426c16db','2015-10-15T22:14:11']
    ,[775,1520977,'course-v1:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','load_video','3169d758ee2d4262b07f0113df743c42','2015-10-15T22:43:35']
    ,[775,1520977,'course-v1:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','play_video','3169d758ee2d4262b07f0113df743c42','2015-10-15T22:43:40']
    ,[991,2221973,'course-v1:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','play_video','3169d758ee2d4262b07f0113df743c42','2015-10-11T11:13:20']
    ,[101,2122312,'art:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','click_courseware','2121d758ee2d4262b07f0113df743a11','2015-11-10T28:32:10']
    ,[101,2122312,'art:TsinghuaX+70800232X+2015_T2','5f421f644193c2d48c84df42aaf7e48b','close_courseware','2121d758ee2d4262b07f0113df743a11','2015-11-10T28:34:10']
    ] 

raw_log_column_type ={
    'enroll_id'  :'int'
    ,'username'  :'int'
    ,'course_id' :'str'
    ,'session_id':'str'
    ,'action'    :'str'
    ,'object'    :'str'
    ,'time'      :'str'
}
# 行为列表
enroll_ID_dict_include_actionList = {
     '775':['load_video','load_video','play_video']
    ,'991':['play_video']
    ,'101':['click_courseware','close_courseware']
}


# 数值化序列 目前只是为了节约内存
action_category = {
    ##  此分类引用自AAAI19那篇论文的代码
     'video_action'   :['seek_video','play_video','pause_video','stop_video','load_video']
    ,'problem_action' :['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
    ,'forum_action'   :['create_thread','create_comment','delete_thread','delete_comment']
    ,'click_action'   :['click_info','click_courseware','click_about','click_forum','click_progress']
    ,'close_action'   :['close_courseware']}
replace_dict = {
    ## 使用数值替换字符串
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
                    ,'create_thread':31 # n
                    ,'create_comment':32 # n
                    ,'delete_thread':33 # n
                    ,'delete_comment':34 #n
                    # click
                    ,'click_info':41
                    ,'click_courseware':42
                    ,'click_about':43
                    ,'click_forum':44 #n
                    ,'click_progress':45 #n
                    ,'close_courseware':46}
                
enroll_ID_dict_include_intList = {
     '775':[15,15,12]
    ,'991':[12]
    ,'101':[42,46]
}
# 词计数结果 仅用于聚类 
## 此处仅用于展示  15，12，42，46四种行为就对应了一个四维的向量
## 真实的模型里有21种行为 所以是21维向量
list_include_all_action_count_vector= [
    [0,0,1,2]  
    ,[0,0,1,0]
    ,[1,1,0,0]]

# 模型输出结果
## 输入
input= [15,15,12] 
# define generate_len = 2
output = [15,12]
