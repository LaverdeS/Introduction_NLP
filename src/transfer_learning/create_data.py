import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/train_data_A.txt', '\t', encoding='mac_roman')
    df_a = df[df['Target'] == 'Atheism']
    df_c = df[df['Target'] == 'Climate Change is a Real Concern']
    df_f = df[df['Target'] == 'Feminist Movement']
    df_h = df[df['Target'] == 'Hillary Clinton']
    df_l = df[df['Target'] == 'Legalization of Abortion']
    dfs = [df_a, df_c, df_f, df_h, df_l]
    dfss = ['df_atheism', 'df_climate', 'df_feminist', 'df_hillary', 'df_legabortion']

    for dataframe, filename_out in zip(dfs, dfss):
      print(dataframe.head(2))
      dataframe.drop(['Target'], axis=1, inplace=True)
      dataframe.columns = ['index', 'text', 'label']
      print(f"number of rows: {dataframe.shape[0]}, 90%head: {round(dataframe.shape[0]*0.9)}")
      train = dataframe
      print(f'train.shape: {train.shape}')
      train.to_csv('./data/'+filename_out + '.csv', encoding='utf-8', index=False)
      print(dataframe.head(2))

    print(df_l.head(2))
    print(df_c.head(2))