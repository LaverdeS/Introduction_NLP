import argparse
import os
from nltk import word_tokenize, RegexpTokenizer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

os.makedirs('./stance/output/')
os.makedirs('./tfidf/output/')

def pre_process(path):
    # preprocessing
    df = pd.read_csv(path, '\t', encoding='mac_roman')
    tokenizer = RegexpTokenizer(r'\w+')
    pattern = '[^A-Za-z]+'
    pd_list = df['Tweet'].apply(lambda x: re.sub(pattern, ' ', x))
    pd_list = pd_list.apply(lambda x: tokenizer.tokenize(x.lower()))
    stances = df['Stance'].apply(lambda x: tokenizer.tokenize(x.lower()))
    targets_out = list(set(df['Target']))
    tweet_stance_out = pd.concat([pd_list, stances], axis=1, sort=False)
    return targets_out, tweet_stance_out, df


def feature_eng(df_target, out_path):
    # Feature Engineering
    print(out_path)
    vectorizer = CountVectorizer()
    bow = list()
    tweet_list = [' '.join(x) for x in df_target['Tweet']]
    for tweet in tweet_list:
        tweet_set = set(tweet.split(" "))
        bow.append(tweet_set)
        vocabulary = [item for tweet_set in bow for item in tweet_set]
    numOfWords = dict.fromkeys(vocabulary, 0)
    for word in vocabulary:
        numOfWords[word] += 1
    dic = set(vocabulary)

    TfidfTransformer(smooth_idf=True, use_idf=True)
    stance = df_target['Stance']
    # Tf-IDF
    word_count_vector = vectorizer.fit_transform(tweet_list)
    feature_names = vectorizer.get_feature_names()
    data_frame = pd.DataFrame(word_count_vector.T.todense(), index=feature_names)
    transposed_cv = data_frame.transpose()
    tfidf = transposed_cv.to_csv('tfidf/'+out_path, sep='\t', encoding='utf-8')
    stance = stance.to_csv('stance/'+out_path, sep='\t', encoding='utf-8')
    return tfidf, stance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/semeval2016-task6-trainingdata.txt')
    parser.add_argument('--out_path', type=str, default='output/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(f'run with params: {args}\n')
    targets, tweet_stance, df = pre_process(args.in_path)

    for target in targets:
        target_bool = df['Target'] == target
        df_target_ = tweet_stance[target_bool]
        out_path_ = args.out_path+target+'.tsv'
        print(out_path_)
        tfidf, stance = feature_eng(df_target=df_target_, out_path=out_path_)
