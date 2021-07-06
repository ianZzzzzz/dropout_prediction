#%%
static = False
binning=True
measure=False
feature= 'course_amount'
data_dict = dict_data_test

dict_ori_and_predict_label = None
measure = False
columns=[
        'Long_interval_mean','Long_interval_var','Long_interval_skew','Long_interval_kurtosis',
        'Short_interval_mean','Short_interval_var','Short_interval_skew','Short_interval_kurtosis',
        
        'video-video','video-answer','video-comment','video-courseware',
        'answer-video','answer—answer','answer-comment','answer-courseware',
        'comment-video','comment-answer','comment-comment','comment-courseware',
        'courseware-video','courseware-answer','courseware-comment','courseware-courseware',
        
        'gender','birth_year' ,'edu_degree',
        'course_category','course_type','course_duration',
        'course_amount','dropout_rate_of_course',
        'student_amount',' dropout_rate_of_user']

feature_location = columns.index(feature)

category_dict = {}

total_sample_number = len(data_dict)

c = 0

if binning ==True:# for continues values feature
    list_feature_values = []
    for e_id,dict_ in data_dict.items():
        data = [*dict_['log_features'],*dict_['info_features']]
        feature_value = data[feature_location]
        
        if np.isnan(feature_value) :
            pass
        else:
            feature_value = round(float(feature_value),2)
        
        list_feature_values.append([e_id,feature_value])

    np_feature_values= np.array(list_feature_values,dtype='float')[:,1]
    np_feature_eid   = np.array(list_feature_values,dtype='float')[:,0] 

    np_feat_cate = pd.cut(
        np_feature_values,
        bins= 10,
        #precision= 1
        labels= False,
        )
# %%
