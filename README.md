20201018 zhengyh186@gmail.com
这两周的工作都围绕AAAI19-Understanding-Droupout-In-Moocs这篇论文展开。
我在以下解读中用到的数据与变量名与论文代码中保持一致，建议配着论文看。
解读如下：

1.作者对数据下如下定义：
(1)C  courses set
(2)U  users set
(3)E  enroll_id  代表一个用户和一门课程的唯一对应 i.e. { [u,c] ....} 

(4)X (u,c） 用户u在课程c下的进行的所有操作 
操作分为 video_action, problem_action等计次数据，先在extract_feat.py中提取，再在main.py中进一步计算统计值。

(5)Z(u,c)	所有用户和课程的原始信息，分为两类：
①分类型：地区、性别、课程类别、用户集群（利用用户间选课情况的余弦相似性计算，相似性大于0.8则聚为一类）等
②连续值型：年龄等
这部分信息在preprocess.py中被提取，并被填充到二维表中以供后续提取。

2.预测模型建立流程如下：
①处理操作数据x
(1)特征增强
1)xi部分：对该注册号下所有日志数据基于 enroll_id进行汇总，统计各种操作的计次值，再求各组数值的统计量（代码中只算了max和mean）。
2）gu 部分：找出注册号对应的那名用户名下，所有登记的课程，与xi		一样计算统计值。
3）gc部分：找出注册号对应的那门课程名下，所有登记的用户，与xi		一样计算统计值。
4）最后在一张二维表里使用merge拼接在一起，论文中对合并操作公式上使用的是抑或运算符，但他代码只是简单拼接，而且这里用抑或好像也没什么意义。
	(2)特征嵌入 论文写的是embedding ，我感觉就是重新编码让特征矩阵更稠密
				代码这部分还在看
	(3)特征融合 步长为mg的卷积层，论文提到这一步作用类似计算统计值，我	还得再看看代码，确认一下。
	
② 处理 原始信息Z 
1)上文提到Z被塞到二维表里，这里对二维表做embedding，我觉得还是为了压缩数据使特征稠密，再输入全连接网络，这里的全连接网络我觉得起到编码器的作用，但是为什么要先用embedding再用全连接网络还有待查明。
③  Attention模块
1)我理解到的是根据Z中每个用户的选课数量，来分配对应用户X特征在最后特征整合那一步的权重。
2) 
④ Weight Sum模块
1)以下是我的假设：
a.论文里假设每个用户注意力资源 是一样的，
	以此来通过Z中每个用户的选课数量来计算分配到每门课程的注意力，进	而在网络中调整用户行为信息Xi的权重。
b.每个用户注意力资源（或者是精力）一致这个假设主观上我觉得站不住脚，但是统计上是不是这样我不能确定，所以可以在用户选课前进行一个类似智力测试的东西，模糊的区分出天才和不怎么聪明的用户，对预测准确度可能会有提升。
c.这部分代码我还没看完
	
⑤预测模块 
1)使用普通的深度学习网络，损失函数选用分类交叉熵
2)这部分代码还没看完，不过看到他在代码里用了BN层来让隐含层与隐含层之间流动的数据分布更均匀。
	
⑥实验部分
1)该模型比XGBoost准确度高零点几，但是算力开销部分没有给出数据。
2)在理解XGBoost的原理。
--------------------------------------------------------------

This is the code for dropout prediction in our AAAI'19 paper:

Wenzheng Feng, Jie Tang, Tracy Xiao Liu, Shuhuai Zhang, Jian Guan. [Understanding Dropouts in MOOCs](http://keg.cs.tsinghua.edu.cn/jietang/publications/AAAI19-Feng-dropout-moocs.pdf). In Proceedings of the 33rd AAAI Conference on Artificial Intelligence (AAAI'19).



## How to run

```bash
# download data from www.moocdata.org
sh dump_data.sh

# extract basic activity features from log file
python feat_extract.py

# integrate different types of features
python preprocess.py

# run CFIN model
python main.py
```
Note: Features used in this demo are less than what we used in the paper, so the performance will be slightly lower than reported score.

# Reference
```
@inproceedings{feng2019dropout,
title={Understanding Dropouts in MOOCs},
author={Wenzheng Feng and Jie Tang and Tracy Xiao Liu and Shuhuai Zhang and Jian Guan},
booktitle={Proceedings of the 33rd AAAI Conference on Artificial Intelligence},
year={2019}
}
```
