
#%%
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

#%%
std=StandardScaler()
mm = MinMaxScaler()
si = SimpleImputer(np.nan,'most_frequent')
pipeline=make_pipeline(
    si,
   std,
    mm)


#%%
train_data_ = pipeline.fit_transform(train_data)
test_data_ = pipeline.fit_transform(test_data)
#%%

train_data =np.array( list_data_train)
test_data  =np.array( list_data_test)

train_label = list_label_train
test_label  = list_label_test
#%%
import xgboost as xgb
dtrain = xgb.DMatrix(train_data, train_label)
dtest = xgb.DMatrix(test_data,test_label)
# 迭代次数，对于分类问题，每个类别的迭代次数，所以总的基学习器的个数 = 迭代次数*类别个数
num_rounds = 20
params = {'max_depth':10
    ,'verbosity ' : 1
    ,'min_child_weight ':0.1
    , 'eta':0.1
    ,'eval_metric':'error'
    ,'objective' : 'binary:logistic'
     }

params = {}
watchlist = [(dtrain,'train'),(dtest,'test')]
model_ = xgb.train(
    params, 
    dtrain, 
    num_rounds,
    watchlist) 

predict_label = model_.predict(dtest)


def predict_label_to_int(predict_label_list,threshold):

    predict_label_int = []
    for i in predict_label:
        value= i
        if value >threshold:label_ = int(1)
        else:label_ = int(0)
        predict_label_int.append(label_)

    return predict_label_int

predict_label_int = predict_label_to_int(predict_label,threshold=0.1)

#%%

f1 = f1_score(predict_label_int,test_label.tolist())
accuracy = accuracy_score(predict_label_int,test_label.tolist())
AUC = roc_auc_score(predict_label_int,test_label)
# precision = precision_score(predict_label_int,test_label.tolist())
# recall = recall_score(predict_label_int,test_label.tolist())

print(f1,AUC)



# %%
def xgboost_sklearn_style_api(
    train_data:ndarray,
    train_label,
    test_data,
    test_label)->None:
    from xgboost import XGBClassifier # sklearn style api   
    params = {}
    XGB = XGBClassifier(
        learning_rate= 0.05,
        max_depth=10,
        random_state=1)
    model = XGB.fit(
        X=train_data,
        y=train_label)

    from sklearn import metrics
    predict_label = model.predict(test_data_)
    print(
        "Accuracy : %.4g" % metrics.accuracy_score(test_label,predict_label)
    )

#%%
def count_nan(data):
    dict_ = {}

    for row in data:

        for i in range(len(row)):
            if np.isnan(row[i]) :
                try:
                    dict_[i]+=1
                except:
                    dict_[i] = 1
    return dict_
nan_in_test = count_nan(test_data.tolist())
nan_in_train = count_nan(train_data.tolist())

# %%
dict_log_train = json.load(open('after_processed_data_file\\train\\dict_train_log_ordered.json','r'))
list_e_id = list(dict_log_train.keys())

if len(list_e_id) == len(train_data):print('length match! ')
else:print('ERROR LENGTH','ID LENGTH :',len(list_e_id,'\n DATA LENGTH :',len(train_data)))

# %%
dict_scene = {}
for i in range(len(list_e_id)):
    vect = train_data[i]
    scenes = vect[8:24]
    dict_scene[list_e_id[i]] = scenes


# %%
from sklearn.cluster import KMeans
np_scene = np.array(list(dict_scene.values()))
cluster = KMeans(n_clusters=3, random_state=1).fit_predict(np_scene)

# %%
cluster_2 = KMeans(n_clusters=3, random_state=1).fit_predict(np_scene[cluster == 0])

# %%
def load(
                log_path: str,
                read_mode: str,
                return_mode: str,
                encoding_='utf-8',
                columns=None) :
                '''读取csv文件 返回numpy数组'''
                #if read_mode == 'cudf':import cudf as pd
                if read_mode == 'pandas' :
                    import pandas as pd
                  # read full file
                    print('        Start loading ',log_path)
                    log = pd.read_csv(
                        log_path
                        ,encoding=encoding_
                        ,names=columns
                        ,low_memory=False)

                    
                    print('          Total length :',len(log),'rows')
                if return_mode == 'df':return log
                if return_mode == 'values':return log.values

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[1:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

c_info_path = 'raw_data_file\\course_info.csv'
u_info_path = 'raw_data_file\\user_info.csv'
u_info_col = ['user_id','gender','education_degree','birth_year']
c_info_col = ['id','course_id','start_time','end_time','course_type','course_category']

#%%
U_INFO_NP = load(
    log_path =u_info_path,
    read_mode ='pandas',
    return_mode = 'values',
    encoding_ = 'utf-8',
    columns =u_info_col)

#%%
C_INFO_NP = load(log_path =c_info_path,

                read_mode ='pandas',
                return_mode = 'df',
                encoding_ = 'utf-8',
                columns =c_info_col
                )

def load(
            log_path: str,
            return_mode='values',
            read_mode='pandas',
            encoding_='utf-8',
            columns=None,
            test=False) :
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
                    
                    print('    Start loading :',log_path)
                    log = pd.read_csv(
                        log_path
                        ,encoding=encoding_
                        ,names=columns)
                    print('      Total length : ',len(log),'rows.')
                    
                
            if return_mode == 'df':return log
            if return_mode == 'values':return log.values
