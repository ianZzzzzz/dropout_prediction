```mermaid
graph TB
    s1(prediction_log / train_log.csv)
    s2(prediction_log / train_truth.csv)
    s3(course_info.csv)
    s4(user_info.csv)

    ss1(cluster/user_dict)
    ss2(cluster/label_5_10time.npy)
    ss1--load-->p2  
    ss2--load-->p2

    f1(train_features.csv)
    f2(act_feats.pkl)
    f3(train_feat.csv)

    p1[feat_extract.py] 
    p2[preprocess.py]
    p3[main.py]

    s1--read   -->p1
    s2--read   -->p1
    p1--write -->f1 
    s3--read-->p2
    s4--read-->p2
    f1--read   -->p2
    p2--dump-->f2
    p2--write-->f3
    f2--load-->p3
    f3--read -->p3

    
   
```


