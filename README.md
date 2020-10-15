20201015 郑义恒 zhengyh186@gmail.com
我根据论文内容对代码做了相应标注:
    1.论文中X(u,c)的原始行为数据在preprocess.py 中被提取 X^不在此文件中处理
    2.

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
