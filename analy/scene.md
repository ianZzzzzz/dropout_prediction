
# video
    'seek': '11','play':12,'pause':13,
    'stop':14,'load':15
# problem
    'get':21,'check':22,'save':23,'reset_problem':24,'check_correct':25, 'check_incorrect':26
# comment
    'create_thread':31 # n,'create_comment':32 # n,'delete_thread':33 # n,'delete_comment':34 
# click
    'click_info':41,'click_courseware':42,'click_about':43,'click_forum':44 ,'click_progress':45,'close_courseware':46

1214141414141414141414141414 


# mode
## video
    'play'<->'pause':13,12,13,12
    'play'<->'stop' :12,14,12,14
    'play'->'seek'->'pause' 
## 刷课
    'play'  'play' 'play' :12,12,12...
## 挑选课程
    42,46,42,15,12,13 的循环模式
    ->查看简介-关闭简介-查看简介-载入播放视频-暂停-查看简介->
## 不断点击讨论
    44，44，44，44， 
    1 查找答案
    2 查看讨论状况以确定课程情况
