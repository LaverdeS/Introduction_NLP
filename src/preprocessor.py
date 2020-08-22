import argparse
import os
from nltk import word_tokenize, RegexpTokenizer, sent_tokenize
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

porter = PorterStemmer()

if not os.path.exists(os.path.dirname('./stance/output/')):
    os.makedirs('./stance/output/')
if not os.path.exists(os.path.dirname('./tfidf/output/')):
    os.makedirs('./tfidf/output/')


def pre_process(in_path, remove_numbers, remove_special_characters,  remove_stopwords, stem):
    df = pd.read_csv(in_path, '\t', encoding='mac_roman')
    args_list = [remove_numbers, remove_special_characters,  remove_stopwords, stem]
    print(args_list)

    if remove_numbers == 'True':
        print('\nremoving numbers' + remove_numbers)
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub('[0-9]+', ' ', x.lower()))
        print(df.head(3))
        print("tokenizing...")
        df['Tweet'] = [word_tokenize(tweet) for tweet in df['Tweet']]
    else:
        print('\nNot removing numbers' + remove_numbers)

    if remove_special_characters == 'True':
        print('\nremoving special characters...')
        df['Tweet'] = df['Tweet'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', str(x)))
        df['Tweet'] = [word_tokenize(tweet) for tweet in df['Tweet']]
        print(df.head(3))
    else:
        print('not removing special characters' + remove_special_characters)

    if remove_stopwords == 'True':
        print('removing stopwords')
        stop_words = set(stopwords.words('english'))
        df['Tweet'] = [sent_tokenize(str(tweet)) for tweet in df['Tweet']]
        print(df.head(3))

        for idx, tweet in df['Tweet'].items():
            for ix, w in enumerate(tweet):
                tweet[ix] = w.lower() if w not in stop_words else tweet.remove(w)
            df['Tweet'][idx] = list(filter(None, tweet))
        print(df.head(3))
    else:
        print('not removing stopwords because it is set to False or Remove specil characters is set to False' + remove_stopwords)

    if stem and remove_special_characters == 'True':
        print('\nstemming...')
        for tweet in df['Tweet']:
            for ix, w in enumerate(tweet):
                tweet[ix] = porter.stem(w)
        print(df.head(3))
    else:
        print('not stemming...')

    targets_out = list(set(df['Target']))
    tweet_stance_out = df[["Tweet", "Stance"]]
    return targets_out, tweet_stance_out, df


def feature_eng(df, remove_stopwords, stem, out_path):
    vectorizer = CountVectorizer()
    bow = list()
    tweet_list = [''.join(x) for x in df['Tweet']]
    # print(tweet_list)
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
    tfidf = transposed_cv.to_csv(out_path + 'tfidf.tsv', sep='\t', encoding='utf-8')
    stance = stance.to_csv(out_path + 'stance.tsv', sep='\t', encoding='utf-8')
    print("saved to the output path...")
    return tfidf, stance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='F:/NLP/Project/Introduction_NLP/data/train_data_A.txt')
    parser.add_argument('--out_path', type=str, default='F:/NLP/Project/Introduction_NLP/output/')
    parser.add_argument('--remove_numbers', type=str, default='True')
    parser.add_argument('--remove_special_characters', type=str, default='True')
    parser.add_argument('--remove_stopwords', type=str, default='True')
    parser.add_argument('--stem', type=str, default='True')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(f'run with params: {args}\n')
    targets, tweet_stance, df = pre_process(args.in_path, args.remove_numbers,
                                            args.remove_special_characters, args.remove_stopwords, args.stem)

    for target in targets:
        target_bool = df['Target'] == target
        df_ = tweet_stance[target_bool]
        target_str = target.replace(' ', '') + '_'
        out_path_ = args.out_path + target_str
        # print(out_path_)
        tfidf, stance = feature_eng(df=df_, remove_stopwords=args.remove_stopwords, stem=args.stem, out_path=out_path_)
