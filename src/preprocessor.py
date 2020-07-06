from nltk import word_tokenize, RegexpTokenizer
import pandas as pd
import re
from tqdm import tqdm_notebook as tqdm
import string
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import csv



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=10)
    parser.add_argument('--in_path', type=str, default='./semeval2016-task6-trainingdata.txt')
    parser.add_argument('--out_path', type=str, default='./')
    args = parser.parse_args()
    return args

if _name_ == "_main_":
    args = get_args()

    #preprocessing
    path = args.emb_dim
    df = pd.read_csv(path, '\t', encoding='mac_roman')
    pd_list=df[['Tweet','Stance']].values.tolist()
    tokenizer=RegexpTokenizer(r'\w+')
    pattern = '[^A-Za-z]+'
    pd_list=df['Tweet'].apply(lambda x: re.sub(pattern, ' ', x))
    pd_list=pd_list.apply(lambda x: tokenizer.tokenize(x.lower()))
    stances=df['Stance'].apply(lambda x: tokenizer.tokenize(x.lower()))
    targets=list(set(df['Target']))
    tweet_stance=pd.concat([pd_list,stances], axis=1, sort=False)

    search="Climate Change is a Real Concern"
    for target in targets:
      target = df['Target'] == search
      df_target=tweet_stance[target]
    print(df_target)


    #Feature Engineering
    tweet_list=[]
    bow = list()
    word=list()
    tweet_list = [' '.join(x) for x in df_target['Tweet']]
    for tweet in tweet_list:
      tweet_set=set(tweet.split(" "))
      bow.append(tweet_set)
      vocabulary = [item for tweet_set in bow for item in tweet_set]
    numOfWords = dict.fromkeys(vocabulary, 0)
    for word in vocabulary:
        numOfWords[word] += 1
    bow=vocabulary
    dic=set(vocabulary)
    dict_list=list(dic)

    vectorizer = CountVectorizer()
    tf_idf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    stance=df_target['Stance']

    #Bag Of Words
    bag_of_words=vectorizer.fit_transform(tweet_list).toarray()

    #Tf-IDF
    word_count_vector=vectorizer.fit_transform(tweet_list)
    feature_names = vectorizer.get_feature_names()
    data_frame = pd.DataFrame(word_count_vector.T.todense(), index=feature_names)
    transposed_cv=data_frame.transpose()
    transposed_cv.to_csv(args.outpath+'climate_tfidf.tsv', sep='\t', encoding='utf-8')
    stance.to_csv(args.outpath+'climate_stance.tsv', sep='\t', encoding='utf-8')




