import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--cat', type=str, default='')
    parser.add_argument('--dir_out', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    inputf = get_args()
    df = pd.read_csv(inputf.file, encoding='utf-8', header=None)
    x = df.iloc[1:, 0:2]
    y = df.iloc[1:, 2]
    x.columns = ['index', 'text']
    y.columns = ['label']
    # print(f"x:{x.head(25)}, {x.shape}, \ny:{y.head(25)}, {y.shape}")
    x_sp = x.text.tolist()
    y_sp = y.tolist()
    X_train, X_test, y_train, y_test = train_test_split(x_sp, y_sp, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)

    print(f'df({df.shape[0]}), X_train({X_train.shape[0]}), X_test({X_test.shape[0]}), y_train({y_train.shape[0]}), y_test({y_test.shape[0]})')
    X = pd.concat([X_train, y_train], axis=1)
    X.insert(loc=0, column='index', value=np.arange(len(X)) + 1)
    X_test.insert(loc=0, column='index', value=np.arange(len(X_test)) + 1)
    X.columns = ['index', 'text', 'label']
    X_test.columns = ['index', 'text']
    y_train.columns = ['label']
    y_test.columns = ['label']
    print('X.shape: ', X.shape)
    X = X.sample(frac=1)
    train = X.head(round(X.shape[0]*0.9))
    val = X.tail(round(X.shape[0] * 0.1))
    for data, fname in zip([train, val, X_test, y_test], ['train', 'val', 'X_test', 'y_test']):
        data.to_csv(inputf.dir_out+fname+inputf.cat+'.csv', encoding='utf-8', index=False)
    print('done...')