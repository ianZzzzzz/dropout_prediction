'''     from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_classif

        # Vectorization parameters
        # Range (inclusive) of n-gram sizes for tokenizing text.
        NGRAM_RANGE = (1, 2)

        # Limit on the number of features. We use the top 20K features.
        TOP_K = 20000

        # Whether text should be split into word or character n-grams.
        # One of 'word', 'char'.
        TOKEN_MODE = 'word'

        # Minimum document/corpus frequency below which a token will be discarded.
        MIN_DOCUMENT_FREQUENCY = 2

        def ngram_vectorize(train_texts, train_labels, val_texts):
            """Vectorizes texts as n-gram vectors.

            1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

            # Arguments
                train_texts: list, training text strings.
                train_labels: np.ndarray, training labels.
                val_texts: list, validation text strings.

            # Returns
                x_train, x_val: vectorized training and validation texts
            """
            # Create keyword arguments to pass to the 'tf-idf' vectorizer.
            kwargs = {
                    'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
                    'dtype': 'int32',
                    'strip_accents': 'unicode',
                    'decode_error': 'replace',
                    'analyzer': TOKEN_MODE,  # Split text into word tokens.
                    'min_df': MIN_DOCUMENT_FREQUENCY,
            }
            vectorizer = TfidfVectorizer(**kwargs)

            # Learn vocabulary from training texts and vectorize training texts.
            x_train = vectorizer.fit_transform(train_texts)

            # Vectorize validation texts.
            x_val = vectorizer.transform(val_texts)

            # Select top 'k' of the vectorized features.
            selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
            selector.fit(x_train, train_labels)
            x_train = selector.transform(x_train).astype('float32')
            x_val = selector.transform(x_val).astype('float32')
            return x_train, x_val

        train_texts = ['hello how are you','fuck off','I love you']
        train_labels= np.array([1,0,1],dtype = 'bool')
        val_texts = ['how are you','love ','love you']

        x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)
        print(x_train, 'val \n',x_val) 
'''
# Kmean 文本聚类
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')



# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2')
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()


# #############################################################################
# Do the actual clustering

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
