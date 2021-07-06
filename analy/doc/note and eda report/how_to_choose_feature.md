# kdd 1 KEY

        
    对每个用户的序列 分别以日、章节、事件进行采样 resample

    统计用户注册的课程数量、 比例 
        用户的平均行为间隔 
        行为数量
    学生每天花在课程上的时间
        日志的最后几天中学生每天花在课程上的时间
    学生活动在其整个时间序列上的分布情况
        (计算首末次行为时间的中点 以及 平均总耗时
    行为间距大小
    学生行为序列总用时
        学生初次/末次行为与开课/结课时间的间隙大小
    学生活动在开课时间区间内的分布情况
            整体分布
            结课前 5-10 天的分布
    平均时间间隔
##  course features
    平均完成用时
    ● course_id
    ● first log time
    ● enrollment counts !
    ● unique log counts ?
    ● mean time interval !
##  enrollment count features
    学生活动在开课时间区间内的分布情况
        整体分布
        结课前 5-10 天的分布
    ● log counts
    ● unique log counts ?
    ● ratio between unique log counts over log counts
    ● unique log counts by event (nagivate, access,
    problem, video, page_close, discussion, wiki)
    ● unique log counts before end of course (5 days, 10
    days and 30 days before)     !
    ● sequence number of enrollment in that course  !
## enrollment time stats 
    学生行为序列总用时
    学生初次/末次行为与开课/结课时间的间隙大小

    ● log time stats (min, mean, max) !
    ● gap between first and last log of enrollment  !
    ● gap between enrollment first log and course first log !
    ● gap between enrollment last log and course last logs !
    学生活动在其整个时间序列上的分布情况
    (计算首末次行为时间的中点 以及 平均总耗时)
    ● difference between mean log time and mid point between first and last log !
    行为间距大小
    ● log interval stats (mean, 90, 99 and 100 quantiles) !
## Enrollment entropy features
    学生每天花在课程上的时间
    日志的最后几天中学生每天花在课程上的时间
    enrollment entropy over
        ● days
        ● weekdays
        ● fraction (4) of weekdays
        ● hours of the day
        ● hours of the day for the last 1/3/7 days before last logs
        ● object (when event == problem)
        ● chapter ids
## Enrollment sequence features
    ● for each enrollment_id, built sequences of
        ○ weekdays
        ○ objects
            ■ all objects / 'problem' and 'video' objects only
        ○ events
    ● treated sequences as 4 text variables. Ran for each
        ○ svd on 3 grams => first 10 components
        ○ DataRobot stacked predictions from logistic regr.
        & Nystroem SVM on (tuned) n-grams
## user count features and time
    统计用户注册的课程数量、 比例
    用户的平均行为间隔
    行为数量
    stats
    ● enrollment count
    ● binary indicator whether user signed up for each of the 38 courses
    ● unique log count
    ● mean log time interval
    ● sequence number of enrollment for that user
## User entropy features ？
    user entropy over
    ● days
    ● weekdays
    ● fraction (4) of weekdays
    ● hours of the day
## User sequence features
    对每个用户的序列 分别以日、章节、事件进行采样
    ● for each user, built sequences of
        ○ weekdays
        ○ chapter_ids
        ○ events
    ● treated them as 3 text variables. Ran
        ○ SVD on 3 grams => first 10 components
        ○ DataRobot stacked predictions from logistic regr.
        + Nystroem SVM on (tuned) n-grams
  
# kdd 2 key point
## 场景间的转移概率
    in Feature Extractions(4)