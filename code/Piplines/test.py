#%%
import json
#import numpy
#from numpy import DataFrame#
RANDOM_STATE = 1
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

        from xgboost import XGBClassifier # sklearn style api   
        params = {}
        XGB = XGBClassifier(params)
        
        self.model = XGB.fit(
            X=self.data,
            y=self.label)

        
    def predict(self):
        from sklearn import metrics
        predict_label = self.model.predict(data)
        true_label = self.label
        print(
            "Accuracy : %.4g" % metrics.accuracy_score(true_label,predict_label)
        )
        
    def save_model(self,path: str):
        pass
    def load_model(self,path: str):
        pass

    def __str__(self):
        return ' Name : {}, Age : {}, Gender : {} '.format(
            self.mode,self.data,self.label)
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


#%%
test_label = json.load(open(
    'Final_Dataset\\test_label.json','r'))
test_data = json.load(open(
   'Final_Dataset\\test_data.json','r'))
train_label = json.load(open(
    'Final_Dataset\\train_label.json','r'))
train_data = json.load(open(
    'Final_Dataset\\train_data.json','r'))
df_test_data ,df_test_label = to_df(
    test_data,test_label)
df_train_data ,df_train_label = to_df(
    train_data,train_label)

#%%  

# train
xgb_model = Model.train(
    mode='xgb',
    data = df_train_data,
    label= df_train_label
    #show_info = True
    )
# xgb is training 
# measurment： accuracy
# interactive plot

xgb_model.predict(
    data  = df_test_data,
    label = df_test_label)

#%%
# use xgb-model pridict     
# measurment：mae
# interactive plot

# save
xgb_model.save_model(path = '')
# load model
xgb_model = Model.load_model(path = '')

    
# %%

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:28:29 2017
@author: wyl
original link: https://www.yanlongwang.net/Python/python-interactive-mode/
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import math
    
plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig=plt.figure()

ax=fig.add_subplot(1,1,1)
ax.axis("equal") #设置图像显示的时候XY轴比例
plt.grid(True) #添加网格

plt.ion()  #interactive mode on

for t in range(10):

    #障碍物船只轨迹
    #obsX=IniObsX+IniObsSpeed*math.sin(IniObsAngle/180*math.pi)*t
    #obsY=IniObsY+IniObsSpeed*math.cos(IniObsAngle/180*math.pi)*t
    
    obsX = t
    obsY = math.cos(t)
    
    ax.scatter(obsX,obsY,c='b',marker='.')  #散点图
    #ax.lines.pop(1)  删除轨迹
    #下面的图,两船的距离
    plt.pause(0.001)


# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(
    fig1, update_line, 25, fargs=(data, l),
    interval=50, blit=True)

# To save the animation, use the command: line_ani.save('lines.mp4')

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)
# To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()