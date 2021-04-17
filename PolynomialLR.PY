
#%%
from sklearn.preprocessing import PolynomialFeatures

from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame
import matplotlib.pyplot as plt

import json
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn import linear_model 
from sklearn import svm
#%%
np.set_printoptions(suppress=True)
#%%

# 数据建模


#%%
XGB_RESULT = assemble_predicted_and_predict_label(
    list_e_id=list_e_id_test,
    mode = 'TorF',
    list_label = list_label_test,
    predict_label= xgb_model(train_data_simple,test_data_simple)
    )


LR_RESULT = assemble_predicted_and_predict_label(
    list_e_id=list_e_id_test,
    list_label = list_label_test,
    mode = 'TorF',
    predict_label= lr_model(train_data_simple_padding,test_data_simple_padding)
    )

#%%
def choose_data_from_dict(mode:str)-> dict:
    path = 'after_processed_data_file\\dict_data_'+mode+'_for_analy_fix_birth.json'
    dict_ = json.load(open(path,'r'))
    new_dict = {}
    for eid,dict__ in dict_.items():
        static = dict__['log_features'][:8]
        infomation = dict__['info_features']
        label = dict__['label']

        new_dict[eid] = {}
        new_dict[eid]['static'] = static
        new_dict[eid]['infomation'] = infomation
        new_dict[eid]['label'] = label

    return new_dict
def assemble_transfor_matrix_and_others(matrix_dict,others_dict)-> dict:
    if len(matrix_dict)!=len(others_dict):print('Error length mismatch.')
    else:
        new_dict = {}
        for eid in others_dict.keys():

            new_dict[eid] = {}
            new_dict[eid]['features'] = [
                                    *others_dict[eid]['static'] ,
                                    *others_dict[eid]['infomation'] ,
                                    *matrix_dict[int(eid)] ]
            new_dict[eid]['label'] = others_dict[eid]['label']

        return new_dict
def prepare_label(dict_data)-> list:
    """
        dict_data : { e_id: {
                            'log_features':[] ,
                            'info_features':[] ,
                            'label' : 1 or 0
                            } 
                    }

    Returns:
        list_assemble_data : [[ *log_features , *info_features ],......]
        list_e_id : [ e_id_1 ,e_id_2,...... ]
        list_label : [ 0, 1, ..........]
    """    
    list_assemble_data = []
    list_e_id = []
    list_label = []
    for e_id,dict_ in dict_data.items():

        log_data  = dict_['features']

        label     = dict_['label']

        list_assemble_data.append(log_data)
        list_e_id.append(int(e_id))
        list_label.append(int(label))

    return list_assemble_data , list_e_id , list_label
def nomilized_matrix(dict_matrix,matrix_type: str,ignore=False)-> dict:
    import numpy as np
    if matrix_type =='complex':
        for eid,dict_ in dict_matrix.items():
            matrix_= dict_['features'][18:]
            sum_ = np.sum(matrix_)
            if (sum_ != 0):
                for i in range(18,502):
                    dict_matrix[eid]['features'][i] =int(30000*dict_matrix[eid]['features'][i]/sum_)
        return dict_matrix
    if matrix_type =='simple':
        new_dict = {}
        for eid,dict_ in dict_matrix.items():
            if ignore ==False:    
                matrix_= dict_['log_features'][8:]           
                sum_ = np.sum(matrix_)
                if sum_ != 0:
                    for i in range(8,24):
                        try:
                            dict_matrix[eid]['log_features'][i] =int(1000*dict_matrix[eid]['log_features'][i]/sum_)
                        except:
                            print(dict_matrix[eid]['log_features'][i],sum_)
                        
                    
            new_dict[eid]={}
            new_dict[eid]['features'] = [
                *dict_matrix[eid]['info_features'],
                *dict_matrix[eid]['log_features']
                ]
           
            new_dict[eid]['label'] = dict_matrix[eid]['label']

        return new_dict
def predict_label_to_int(predict_label_list,threshold):

    predict_label_int = []
    for i in predict_label_list:
        value= i
        if value >threshold:label_ = int(1)
        else:label_ = int(0)
        predict_label_int.append(label_)

    return predict_label_int

def measure(predict_label_int,list_label_test):
    f1             = f1_score(predict_label_int,list_label_test)
    accuracy = accuracy_score(predict_label_int,list_label_test)
   # AUC =       roc_auc_score(predict_label_int,list_label_test)
    precision = precision_score(predict_label_int,list_label_test)
    recall = recall_score(predict_label_int,list_label_test)

    print(
        'F1',round(f1,4),
        'precision',round(precision,2),
        'recall',round(recall,2),
        'ACC',round(accuracy,2),
        )
def plot_AUC(ori_label,predict_label):
    import pylab as plt
    import warnings;warnings.filterwarnings('ignore')
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(ori_label, predict_label)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
def xgb_model(train_data,test_data)->list:
    dtrain = xgb.DMatrix(train_data, list_label_train)
    dtest = xgb.DMatrix(test_data,list_label_test)
    num_rounds = 10
    params = {}
    watchlist = [(dtrain,'train'),(dtest,'test')]
    XGB_nom = xgb.train(
        params, 
        dtrain, 
        num_rounds,
        watchlist,
        early_stopping_rounds=10) 
        
    #import pickle
    #pickle.dump(XGB_nom, open("XGB_simple_no_nom_f1_9015.pickle.dat", "wb"))

    XGB_predict_label = XGB_nom.predict(dtest)
    plot_AUC(list_label_test,XGB_predict_label)
    for i in [0.1,0.2,0.3,0.4,0.5,0.6]:
            
        XGB_predict_label_int = predict_label_to_int(XGB_predict_label,threshold=i)
        #print('XGB ',i,' : ')
        measure(
                XGB_predict_label_int,list_label_test)
    
    #return XGB_predict_label_int

def lr_model(train_data_padding,test_data_padding)->list:
    lr = linear_model.LinearRegression(
        normalize=True
        ,n_jobs=-1)
    lr.fit(train_data_padding, list_label_train)
    result = lr.predict(test_data_padding)

    plot_AUC(list_label_test,result)

    for i in [0.001, 0.499,0.5,0.501 ]:
    #for i in [0.5 ]:
        predict_LinearRegression = predict_label_to_int(
            result,
            threshold= i)
        #print('LR ',i,' : ')
        measure(
            predict_LinearRegression,list_label_test)

   # return predict_LinearRegression

def assemble_predicted_and_predict_label(
    list_label,
    list_e_id,
    predict_label,
    mode='eid')-> dict:
    """
    Returns:
        dict: { e_id:[ predicted_label , predict_label ]}
    """    
    if mode =='eid': 
        dict_ori_and_predict_label = {}
        for row in range(len(list_label)):
            e_id = list_e_id[row]
            ori_label = list_label[row]
            dict_ori_and_predict_label[e_id] = [ori_label,predict_label[row]]

        return dict_ori_and_predict_label
    if mode =='TorF':
        dict_ ={
                'TP':[],
                'TN':[],
                'FP':[],
                'FN':[]
            }
        
        for i in range(len(list_e_id)):
            eid = list_e_id[i]
            ori = list_label[i]
            pre = predict_label[i]
            
            if pre==ori: # SUCCESS
                if ori ==1:
                    dict_['TP'].append(eid)
                if ori ==0:
                    dict_['TN'].append(eid)
            else:        # Fail
                if pre ==1:
                    dict_['FP'].append(eid)
                if pre ==0:
                    dict_['FN'].append(eid)
        return dict_


#%%
# Simple matrix
dict_train_simple_matrix = json.load(open('after_processed_data_file\\dict_data_train_for_analy_fix_birth.json','r'))
dict_test_simple_matrix  = json.load(open('after_processed_data_file\\dict_data_test_for_analy_fix_birth.json','r'))


dict_train_simple_nom = nomilized_matrix(
    dict_train_simple_matrix,
    matrix_type = 'simple',
    ignore=True)
dict_test_simple_nom = nomilized_matrix(
    dict_test_simple_matrix,
    matrix_type = 'simple',
    ignore= True)

list_data_train_simple,list_e_id_train,list_label_train = prepare_label(dict_train_simple_nom)
list_data_test_simple,list_e_id_test,list_label_test    = prepare_label(dict_test_simple_nom)
'''
new_ = []
for i in range(len(list_data_test_simple)):
    new_.append([*list_data_test_simple[i],list_label_train[i]])
import pandas as pd 
pd.DataFrame(new_).to_csv('list_data_train_simple_with_label.csv')
'''
train_data_simple =np.array( list_data_train_simple)
test_data_simple  =np.array( list_data_test_simple)

 
Padding = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data_simple_padding = Padding.fit_transform(train_data_simple)
test_data_simple_padding  = Padding.fit_transform(test_data_simple)
#%%
poly = PolynomialFeatures(degree=2) #degree=2表示二次多项式
train_poly = poly.fit_transform(train_data_simple_padding) #构造datasets_X二次多项式特征X_poly
test_poly = poly.fit_transform(test_data_simple_padding) #构造datasets_X二次多项式特征X_poly


lr_model(train_poly,test_poly)

xgb_model(train_poly,test_poly)

#%% 效果不如 degree 2

#poly_3 = PolynomialFeatures(degree=3) #degree=2表示二次多项式
train_poly_3 = poly_3.fit_transform(train_data_simple_padding) #构造datasets_X二次多项式特征X_poly
test_poly_3 = poly_3.fit_transform(test_data_simple_padding) #构造datasets_X二次多项式特征X_poly


lr_model(train_poly_3,test_poly_3)

xgb_model(train_poly_3,test_poly_3)
# %%


train_data_padding=train_data_simple_padding
test_data_padding =test_data_simple_padding
#from sklearn.svm import LinearSVC
#SVM = LinearSVC()
#SVM = svm
from sklearn.svm import SVC
#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
SVM = SVC(
    max_iter=5,kernel="sigmoid",degree= 2,
  #  C=0.01,gamma=0.001
    )
SVM.fit(X=train_data_padding, y=list_label_train)
result = SVM.predict(test_data_padding)

plot_AUC(list_label_test,result)
measure(result,list_label_test)
# %%
def svm_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC    
    model = SVC(max_iter=5,kernel='sigmoid', probability=True)    
    param_grid = {'C': [1e-2, 1e-1, 1, 10], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)
        print(best_parameters['C'],best_parameters['gamma'])    
    #model = SVC(max_iter=5,kernel='sigmoid', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    #model.fit(train_x, train_y)    
    #return model
#%%
svm_cross_validation(train_data_padding, list_label_train)
# %%
