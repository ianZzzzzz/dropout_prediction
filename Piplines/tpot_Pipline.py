import tpot

from tpot import TPOTClassifier
import json
import numpy as np

test_data = np.array(json.load(open(
	'json_file\\into_model 2.0\\list_test_static_info_dataset_2.0.json','r')
	))

test_label = np.array(json.load(open(
	'json_file\\into_model 2.0\\list_test_label.json','r')
	))

train_data = np.array(json.load(open(
	'json_file\\into_model 2.0\\list_train_static_info_dataset_2.0.json','r')
	))

train_label = np.array(json.load(open(
	'json_file\\into_model 2.0\\list_train_label.json','r')
	))


tpot = TPOTClassifier(
	#generations = 2,
	max_time_mins=10,

	max_eval_time_mins = 0.05, #：浮点型，可选（默认= 5）TPOT必须评估多少分钟才​​能评估一条管道。
	early_stop = 50,
	population_size=40,


	random_state = 1,
	scoring = 'f1',
	verbosity = 3,
	#log_file = open('model_run_log\\tpot_log.txt','w'),
	#memory='auto',
	#warm_start = True
	)
tpot.fit(
	train_data, 
	train_label
	)

tpot.score(
	test_data, 
	test_label
	)
'''
	'DecisionTreeClassifier(
		input_matrix, 
		DecisionTreeClassifier__criterion=gini, 
		DecisionTreeClassifier__max_depth=2, 
		DecisionTreeClassifier__min_samples_leaf=17, 
		DecisionTreeClassifier__min_samples_split=7)': 
	Pipeline(steps=[
		('decisiontreeclassifier',
		DecisionTreeClassifier(
			max_depth=2, 
			min_samples_leaf=17,
			min_samples_split=7, 
			random_state=1)
		)
		]), 
	'GaussianNB(
		MinMaxScaler(
			SelectPercentile(

				OneHotEncoder(
					input_matrix, 
					OneHotEncoder__minimum_fraction=0.2, 
					OneHotEncoder__sparse=False, 
					OneHotEncoder__threshold=10
					), 
				SelectPercentile__percentile=9
				)
				)
				)':
	 Pipeline(steps=[
		('onehotencoder',OneHotEncoder(minimum_fraction=0.2, sparse=False)),
        ('selectpercentile', SelectPercentile(percentile=9)),
        ('minmaxscaler', MinMaxScaler()),   
        ('gaussiannb', GaussianNB())
		])
'''

'''
{
	DecisionTreeClassifier(
		input_matrix, 
		DecisionTreeClassifier__criterion=gini,
		DecisionTreeClassifier__max_depth=2, 
		DecisionTreeClassifier__min_samples_leaf=17, 
		DecisionTreeClassifier__min_samples_split=7
		)
	
	: 
	Pipeline(steps=[('decisiontreeclassifier',
                 DecisionTreeClassifier(max_depth=2, min_samples_leaf=17,
                                        min_samples_split=7, random_state=1))])}
										
	'''