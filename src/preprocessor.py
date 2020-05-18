from nltk import word_tokenize, RegexpTokenizer
import pandas as pd
import re
from tqdm import tqdm_notebook as tqdm
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import pickle

path = 'semeval2016-task6-trainingdata.txt'
df = pd.read_csv(path, '\t', encoding='mac_roman')
pd_list = df['Tweet'].values.tolist()
tokenizer = RegexpTokenizer(r'\w+')
pattern = '[^A-Za-z]+'
pd_list = df['Tweet'].apply(lambda x: re.sub(pattern, ' ', x))
pd_list = pd_list.apply(lambda x: tokenizer.tokenize(x.lower()))
targets = list(set(df['Target']))

for target in targets:
    target = df['Target'] == 'Legalization of Abortion'
    df_target = pd_list[target]
df_target

vectorizer = CountVectorizer()
tweet_list = []
frequency = [Counter(tweet) for tweet in df_target]
set_all = set()
new_list = [' '.join(x) for x in df_target]
vec_transform = vectorizer.fit_transform(new_list)
for tweet in df_target:
    tweet_set = set(tweet)
    set_all = set_all.union(tweet_set)
cv = pd.DataFrame(vec_transform.toarray())
transform = TfidfTransformer()
trans = transform.fit(vec_transform)
cv.to_csv('legalization_of_abortion_bow.tsv', sep='\t', encoding='utf-8')
with open('legalization_of_abortion_tfidf.pkl', 'wb') as f:
    pickle.dump(trans.idf_, f)
