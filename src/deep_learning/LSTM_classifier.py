import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = '1'

import nltk
from tqdm.notebook import tqdm as tqdm
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import cufflinks as cf
import pandas as pd
import random
import csv
import re


nltk.download('stopwords')


def generate_training_df(df, category, min_length=5, verbose=True):
    # df refactor
    df = df.reindex(columns=['ID', 'Stance', 'Tweet', 'Target'])
    columns = ['A', 'label', 'tweet', 'target']
    df.columns = columns
    df = df.drop(columns=['A'])
    preview = random.randint(0, df.shape[0] - 11)
    print("\ndf:\n\n", df[preview:preview + 10]) if verbose else None

    # df per category
    df_cat = df[df.target.str.contains(category, case=False, regex=False) == True]
    print(f"\ndf_{category}:\n\n", df_cat.head()) if verbose else None
    print(f"number of samples in df({category}): ", df_cat.shape[0])

    # labels = [-1,0-1]
    for index, label in enumerate(df_cat.label):
        if label == 'AGAINST':
            df_cat.label[index] = -1
        elif label == 'FAVOR':
            df_cat.label[index] = 1
        else:
            df_cat.label[index] = 0

    # filter tweets with len < min_lenght
    df_catFilter = df_cat[df_cat.tweet.apply(lambda x: len(str(x)) > min_length)]
    print(f"number of samples after filter -> len(tweet) > {min_length}: ", df_catFilter.shape[0])

    # shuffle and refactor
    df_catFilter = df_catFilter.sample(frac=1)
    df_catFilter = df_catFilter.drop(columns=['target'])
    df_catFilter.columns = ['label', 'text']
    df_catFilter = df_catFilter.reindex(columns=['text', 'label'])
    print("finished with category '{}'".format(category))

    return df_catFilter


def generate_training_df_silent(df, category, min_length=0):
    # df refactor
    df = df.reindex(columns=['ID', 'Stance', 'Tweet', 'Target'])
    columns = ['A', 'label', 'tweet', 'target']
    df.columns = columns
    df = df.drop(columns=['A'])
    preview = random.randint(0, df.shape[0] - 11)
    df_cat = df[df.target.str.contains(category, case=False, regex=False) == True]

    for index, label in enumerate(df_cat.label):
        if label == 'AGAINST':
            df_cat.label[index] = -1
        elif label == 'FAVOR':
            df_cat.label[index] = 1
        else:
            df_cat.label[index] = 0

    df_catFilter = df_cat[df_cat.tweet.apply(lambda x: len(str(x)) > min_length)]
    df_catFilter = df_catFilter.sample(frac=1)
    df_catFilter = df_catFilter.drop(columns=['target'])
    df_catFilter.columns = ['label', 'text']

    return df_catFilter


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '').replace('\d+', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


def train_eval_multiclass_LSTM(df, category, verbose=False, silent=False):
    # df_cat = df.copy()
    print(f"generating df for {category}...") if verbose else None
    if silent == False:
        df_cat = generate_training_df(df=df, category=category, min_length=2, verbose=verbose)
    else:
        df_cat = generate_training_df_silent(df=df, category=category, min_length=2)
    df_cat = df_cat.reindex(columns=['text', 'label'])

    print("tokenizing...") if verbose else None
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df_cat['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index)) if verbose else None

    X = tokenizer.texts_to_sequences(df_cat['text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape) if verbose else None
    Y = pd.get_dummies(df_cat['label']).values
    print('Shape of label tensor:', Y.shape) if verbose else None

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print("X_train.shape,Y_train.shape", X_train.shape, Y_train.shape) if verbose else None
    print("X_test.shape,Y_test.shape", X_test.shape, Y_test.shape) if verbose else None

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary()) if silent == False else None

    cp_callback = ModelCheckpoint(filepath='./' + category.replace(' ', '') + '.checkpoint', save_weights_only=True,
                                  verbose=1)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=
    [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001), cp_callback])

    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
    return history


def visualize_history(history):
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show();


def create_LSTM_model(EMBEDDING_DIM, dropout, optimizer):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    global model_count
    model_count += 1
    print(f'\nmodel_count: {model_count}')
    return model


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--emb_dim', type=int, default=10)
#     parser.add_argument('--ws', type=int, default=5)
#     parser.add_argument('--samples', type=int, default=50)
#     parser.add_argument('--in_path', type=str, default='./1000-reviews.txt')
#     parser.add_argument('--out_path', type=str, default='./pca_model_v2.txt')
#
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":

    # CONFIG
    # args = get_args()
    # print(args)
    # generate_pca_model(in_path=args.in_path, out_path=args.out_path, embedding_dimension=args.emb_dim,
    #                    window_size=args.ws, num_samples=args.samples)
    # evaluate(args.out_path)

    # DATA ANALYSIS
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    pd.set_option('mode.chained_assignment', None)

    # read data and summarize
    corpus = 'train_data_A.txt'
    df = pd.read_csv(corpus, header=0, sep='\t', encoding='mac_roman')
    print("\nnumber of samples for this corpus: ", df.shape[0])
    print(f"\n{df.Target.value_counts()}")
    print(df['Target'].value_counts().sort_values(ascending=False))

    # PLOT
    # cufflinks conect to iplot pandas series.
    # df['Target'].value_counts().sort_values(ascending=False).iplot(kind='bar', yTitle='number of Complaints',
    #                                                                title='Number of tweets per target(topic)')

    # PREPROCESSING
    # read data and pre-process it
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    corpus = 'train_data_A.txt'
    df = pd.read_csv(corpus, header=0, sep='\t', encoding='mac_roman')
    start = random.randint(0, df.shape[0] - 11)
    print(df['Tweet'][start:start + 10])
    df['Tweet'] = df['Tweet'].apply(clean_text)
    print('\n', df['Tweet'][start:start + 10])

    # GRID SEARCH
    # parameters
    # static
    MAX_NB_WORDS = 50_000
    MAX_SEQUENCE_LENGTH = 250
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    epochs = 20  # I have earlystopping so I don't need to tune this.

    # category options
    categories = ['Atheism',
                  'Climate Change is a Real Concern',
                  'Feminist movement',
                  'Hillary Clinton',
                  'Legalization of abortion']

    # model params
    batch_size = [1, 8, 32, 64]
    EMBEDDING_DIM = [50, 100, 300]
    dropout = [0.1, 0.3, 0.5, 0.8]
    optimizer = ['Adam', 'SGD', 'Adamax', 'Nadam']
    # add learning rate...

    # pre-processing params
    remove_hash, remove_stopwords, remove_numbers = [[True, False]] * 3
    remove_stopwords = [True, False]
    remove_numbers = [True, False]
    min_lenght = [0, 2, 5, 10, 20, 30]
    # text = text.replace('x', '').replace('\d+', '')

    param_grid = {'EMBEDDING_DIM': EMBEDDING_DIM, 'dropout': dropout, 'optimizer': optimizer}
    print(param_grid)

    # add all of the LSTM parameter from the LSTM_layer in keras as well

    # LSTM MODEL
    model_count = 0

    # for atheism
    df_atheism = generate_training_df_silent(df, categories[0], min_length=0)
    df_atheism = df_atheism.reindex(columns=['text', 'label'])
    print("tokenizing...")  # if verbose else None
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df_atheism['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))  # if verbose else None

    X = tokenizer.texts_to_sequences(df_atheism['text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)  # if verbose else None
    Y = pd.get_dummies(df_atheism['label']).values
    print('Shape of label tensor:', Y.shape)  # if verbose else None

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print("X_train.shape,Y_train.shape", X_train.shape, Y_train.shape)  # if verbose else None
    print("X_test.shape,Y_test.shape", X_test.shape, Y_test.shape)  # if verbose else None

    LSTM_model = KerasClassifier(build_fn=create_LSTM_model, epochs=epochs)
    grid = GridSearchCV(estimator=LSTM_model, param_grid=param_grid, n_jobs=None, cv=3)
    print("grid search built...")

    # EXECUTE_GRID_SEARCH
    # cp_callback = ModelCheckpoint(filepath='./atheism.checkpoint', save_weights_only=True, verbose=1)
    results_per_batchsize = []
    model_count = 0
    for _batch_size in batch_size:
        print(f"\nSTARTING GRIDSEARCH FOR BATCH_SIZE: {_batch_size}")
        grid_result = grid.fit(X_train, Y_train, epochs=epochs, batch_size=_batch_size, validation_split=0.1, callbacks=
        [EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001, restore_best_weights=True)],
                               verbose=1)
        results_per_batchsize.append(grid_result)
        print(f"\n__________________________________________________________________________________________\
              ___________\nbest_estimator for batch_size {_batch_size}: {grid_result.best_estimator}\nbest_score: \
            {grid_result.best_score}\nbest_params: {grid_result.best_params}\n")