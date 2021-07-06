
#%%
col_index = {
    'comment-courseware':19,
    'answer-video':12,
   'comment-video':16,
    'answer-courseware':15,
    'courseware-video':20,
    'video-courseware':11,
    'answer-comment':14,
    'gender':24,
    'short_skew':6,
    'courseware-answer':21,
    'comment-answer':17,
    'video-answer':9,
    'courseware-comment':22,
    'long_skew':2,
    'video-comment':10}

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

col_index = {}
for i in columns:
    col_index[i] = columns.index(i)
#%%
col_index =   {
    'Long_interval_mean': 0,
    'Long_interval_var': 1,
    'Long_interval_skew': 2,
    'Long_interval_kurtosis': 3,
    'Short_interval_mean': 4,
    'Short_interval_var': 5,
    'Short_interval_skew': 6,
    'Short_interval_kurtosis': 7,
    'video-video': 8,
    'video-answer': 9,
    'video-comment': 10,
    'video-courseware': 11,
    'answer-video': 12,
    'answer—answer': 13,
    'answer-comment': 14,
    'answer-courseware': 15,
    'comment-video': 16,
    'comment-answer': 17,
    #'comment-comment': 18,
    'comment-courseware': 19,
    'courseware-video': 20,
    'courseware-answer': 21,
    'courseware-comment': 22,
    'courseware-courseware': 23,
    'gender': 24,
    'birth_year': 25,
    'edu_degree': 26,
    'course_category': 27,
    'course_type': 28,
    'course_duration': 29,
    'course_amount': 30,
    'dropout_rate_of_course': 31,
    'student_amount': 32,
    ' dropout_rate_of_user': 33}

important_feat_name = list(col_index.values())
important_train_data = train_data_padding[:,important_feat_name]

important_test_data = test_data_padding[:,important_feat_name]

model_important_feat = linear_model.LinearRegression()
model_important_feat.fit(important_train_data, list_label_train)


result_lr = model_important_feat.predict(important_test_data)
for i in [0.001, 0.499,0.5,0.501]:
    result = predict_label_to_int(
        result_lr,
        threshold= i)
    measure(
        result,list_label_test)
# %%
