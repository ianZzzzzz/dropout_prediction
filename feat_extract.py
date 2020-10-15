import pandas as pd


# 读取文件 记得把这坨屎封包起来
train = pd.read_csv('prediction_log/train_log.csv')
test = pd.read_csv('prediction_log/test_log.csv')
train_truth = pd.read_csv('prediction_log/train_truth.csv', index_col='enroll_id')
test_truth = pd.read_csv('prediction_log/test_truth.csv', index_col='enroll_id')
#--------------------------------------------------------------------------------------------
all_truth = pd.concat([train_truth, test_truth])
all_log = pd.concat([train, test])
#--------------------------------------------------------------------------------------------
train_enroll = list(set(list(train['enroll_id'])))
test_enroll = list(set(list(test['enroll_id'])))
#--------------------------------------------------------------------------------------------
# 以下 是特征 Z（u,c)中包含的行为数据 分为以下几个大类 再分为list中的小类 
# 此处初始化这些分类的名字
video_action = ['seek_video','play_video','pause_video','stop_video','load_video']
problem_action = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
forum_action = ['create_thread','create_comment','delete_thread','delete_comment']
click_action = ['click_info','click_courseware','click_about','click_forum','click_progress']
close_action = ['close_courseware']
#--------------------------------------------------------------------------------------------
# 对所有行为数据基于注册号进行汇总
# 并统计各个注册号的action总数
all_num = all_log.groupby('enroll_id').count()[['action']]
all_num.columns = ['all#count'] # 'all#count'注册号的action总数量列
# 推测指的是每个课程中有多个session 在日志中每个session都会被多次访问
# 此处只要统计session是否被访问过 而对访问的次数不重视 所以进行去重
session_enroll = all_log[['session_id']].drop_duplicates()
# 计算每个注册号下被访问过的session 总数
session_num = all_log.groupby('enroll_id').count()
all_num['session#count'] = session_num['session_id']
#--------------------------------------------------------------------------------------------

for a in video_action + problem_action + forum_action + click_action + close_action:
    action_ = (all_log['action'] == a).astype(int)
    all_log[a+'#num'] = action_
    action_num = all_log.groupby('enroll_id').sum()[[a+'#num']]
    all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)
all_num = pd.merge(all_num, all_truth, left_index=True, right_index=True)
enroll_info = all_log[['username','course_id','enroll_id']].drop_duplicates()
enroll_info.index = enroll_info['enroll_id']
del enroll_info['enroll_id']
all_num = pd.merge(all_num, enroll_info, left_index=True, right_index=True)
all_num.loc[test_enroll].to_csv('test_features.csv')
all_num.loc[train_enroll].to_csv('train_features.csv')

