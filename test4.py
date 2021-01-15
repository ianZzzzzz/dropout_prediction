'''自监督文本分类
'''
import os

import numpy as np

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing   import FunctionTransformer
from sklearn.linear_model    import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics         import f1_score
def load_split_data():
    data = fetch_20newsgroups(subset='train', categories=None)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))
    print()

    Sample, Label = data.data, data.target
    Sample_train, Sample_test, Label_train, Label_test = train_test_split(Sample, Label)

    dataset = dict(
        train_S= Sample_train
        ,train_L= Label_train
        ,test_S= Sample_test
        ,test_L= Label_test)

    return dataset

def set_parameters():
    sdg_params = dict(alpha=1e-5, penalty='l2', loss='log')
    vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
    return sdg_params,vectorizer_params

def set_pipline():
        # Supervised Pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(**sdg_params)),   ])
    # SelfTraining Pipeline
    st_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        ('clf', SelfTrainingClassifier(SGDClassifier(**sdg_params), verbose=True)),])
    # LabelSpreading Pipeline
    ls_pipeline = Pipeline([
        ('vect', CountVectorizer(**vectorizer_params)),
        ('tfidf', TfidfTransformer()),
        # LabelSpreading does not support dense matrices
        ('todense', FunctionTransformer(lambda x: x.todense())),
        ('clf', LabelSpreading()),])
    
    piplines = dict(ori = pipeline ,self = st_pipeline,ls = ls_pipeline)
    return piplines


def eval_and_print_metrics(
    clf
    , Sample_train, Label_train
    , Sample_test, Label_test
    ):
    print("Number of training samples:", len(Sample_train))
    print("Unlabeled samples in training set:",
          sum(1 for x in Label_train if x == -1))
    clf.fit(Sample_train, Label_train)
    print('fit')
    y_pred = clf.predict(Sample_test)
    print('predict')
    print("Micro-averaged F1 score on test set: "
          "%0.3f" % f1_score(Label_test, y_pred, average='micro'))
    print("-" * 10)
    print()




# 加载并切片数据
dataset = load_split_data()
# 设置模型参数
sdg_params , vectorizer_params    = set_parameters()
# 设置模型流水线
self_sup_pipeline = set_pipline()['self']
# 运行模型打印结果
dataset['train_L'] = np.array(list(map(lambda x : -1,dataset['train_L'])))
dataset['train_L']
print("SelfTrainingClassifier on 20% of the training data (rest "
        "is unlabeled):")
eval_and_print_metrics(
    self_sup_pipeline
    , dataset['train_S']
    , dataset['train_L']
    , dataset['test_S']
    , dataset['test_L'])



print("Supervised SGDClassifier on 100%% of the data:")

# select a mask of 20% of the train dataset
y_mask = np.random.rand(len(Label_train)) < 0.2


# X_20 and y_20 are the subset of the train dataset indicated by the mask
X_20, y_20 = map(list, zip(*((x, y)
                    for x, y, m in zip(Sample_train, Label_train, y_mask) if m)))
print("Supervised SGDClassifier on 20% of the training data:")
eval_and_print_metrics(pipeline, X_20, y_20, Sample_test, Label_test)

# set the non-masked subset to be unlabeled

if 'CI' not in os.environ:
    # LabelSpreading takes too long to run in the online documentation
    print("LabelSpreading on 20% of the data (rest is unlabeled):")
    eval_and_print_metrics(ls_pipeline, Sample_train, Label_train, Sample_test, Label_test)

