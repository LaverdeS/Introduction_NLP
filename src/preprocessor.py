import argparse
import os
from nltk import word_tokenize, RegexpTokenizer, sent_tokenize
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

porter = PorterStemmer()


def pre_process(in_path, remove_numbers, remove_special_characters, remove_stopwords, stem):
    df = pd.read_csv(in_path, '\t', encoding='mac_roman')
    args_list = [remove_numbers, remove_special_characters, remove_stopwords, stem]
    print(args_list)
    print("original string(s): ", df.head(3))

    if remove_numbers == 'True':
        print('\nremoving numbers' + remove_numbers)
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub('[0-9]+', ' ', x.lower()))
        print("tokenizing...")
        df['Tweet'] = [word_tokenize(tweet) for tweet in df['Tweet']]
        print(df['Tweet'].head(3))
    else:
        print('\nNot removing numbers' + remove_numbers)

    if remove_special_characters == 'True':
        print('\nremoving special characters...')
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', str(x)))
        df['Tweet'] = [word_tokenize(tweet) for tweet in df['Tweet']]
        print(df['Tweet'].head(3))
    else:
        print('not removing special characters' + remove_special_characters)

    if remove_stopwords == 'True':
        print('removing stopwords')
        stop_words = set(stopwords.words('english'))
        if remove_special_characters and remove_numbers != 'True':
            df['Tweet'] = [word_tokenize(str(tweet)) for tweet in df['Tweet']]

        for idx, tweet in df['Tweet'].items():
            for ix, w in enumerate(tweet):
                tweet[ix] = w.lower() if w not in stop_words else tweet.remove(w)
            df['Tweet'][idx] = list(filter(None, tweet))
        print(df['Tweet'].head(3))
    else:
        print(
            'not removing stopwords because it is set to False or Remove special characters is set to False' + remove_stopwords)

    if stem == 'True':
        print('\nstemming...')
        if remove_stopwords != 'True':
            df['Tweet'] = [word_tokenize(str(tweet)) for tweet in df['Tweet']]
        for tweet in df['Tweet']:
            for ix, w in enumerate(tweet):
                tweet[ix] = porter.stem(w)
        print(df['Tweet'].head(3))
    else:
        print('not stemming because it is set to False or Remove special characters is set to False')

    print("preprocessing out  -->  ", df['Tweet'].head(3))
    targets_out = list(set(df['Target']))
    tweet_stance_out = df[["Tweet", "Stance"]]
    return targets_out, tweet_stance_out, df

def toy_test():
    print("running toy-test...")
    c1 = ["./train_data_A.txt", "True", "True", "True", "True"]
    c2 = ["./train_data_A.txt", "True", "True", "True", "False"]
    _, df1, _ = pre_process(in_path=c1[0], remove_numbers=c1[1], remove_special_characters=c1[2], remove_stopwords=c1[3], stem=c1[4])
    _, df2, _ = pre_process(in_path=c2[0], remove_numbers=c2[1], remove_special_characters=c2[2], remove_stopwords=c2[3], stem=c2[4])
    print(df1==df2)
    return

def feature_eng(df, remove_stopwords, stem, out_path):
    vectorizer = CountVectorizer()
    bow = list()
    tweet_list = [''.join(x) for x in df['Tweet']]
    for tweet in tweet_list:
        tweet_set = set(tweet.split(" "))
        bow.append(tweet_set)
        vocabulary = [item for tweet_set in bow for item in tweet_set]
    numOfWords = dict.fromkeys(vocabulary, 0)

    TfidfTransformer(smooth_idf=True, use_idf=True)
    stance = df['Stance']
    # Tf-IDF
    word_count_vector = vectorizer.fit_transform(tweet_list)
    feature_names = vectorizer.get_feature_names()
    data_frame = pd.DataFrame(word_count_vector.T.todense(), index=feature_names)
    transposed_cv = data_frame.transpose()
    tfidf = transposed_cv.to_csv(out_path + "_tfidf.tsv", sep='\t', encoding='utf-8')
    stance = stance.to_csv(out_path + "_stance.tsv", sep='\t', encoding='utf-8')
    return tfidf, stance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/test.txt')
    parser.add_argument('--out_path', type=str, default='output/')
    parser.add_argument('--remove_numbers', type=str, default='True')
    parser.add_argument('--remove_special_characters', type=str, default='True')
    parser.add_argument('--remove_stopwords', type=str, default='True')
    parser.add_argument('--stem', type=str, default='True')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # toy_test()

    print(f'run with params: {args}\n')
    targets, tweet_stance, df = pre_process(args.in_path, args.remove_numbers,
                                            args.remove_special_characters, args.remove_stopwords, args.stem)

    for target in targets:
        target_bool = df['Target'] == target
        df_ = tweet_stance[target_bool]
        out_path_ = args.out_path + target.replace(" ", "")
        print(out_path_)
        tfidf, stance = feature_eng(df=df_, remove_stopwords=args.remove_stopwords, stem=args.stem, out_path=out_path_)