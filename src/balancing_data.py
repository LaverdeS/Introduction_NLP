import argparse
import pandas as pd
import random


class balancingData:
    def __init__(self, corpora_path, target_name):
        self.corpora = self.read_csv(corpora_path)
        self.corpora = self.drop_unwanted(self.corpora)
        self.target = self.divide_targets(target_name, self.corpora)
        self.stance_labels_dic = dict()
        self.stance_labels_dic = self.calculate_stance_labels(self.target)
        self.lowest_size = self.stance_labels_dic['lowest_size']
        self.lowest_label_name = self.stance_labels_dic['lowest_label_name']
        self.target_df = pd.DataFrame(columns=['Target','Tweet','Stance'])
        self.append_target_df(target_name)

        # self.against = self.balance_labels('AGAINST', self.lowest_size, self.target, target_name, self.lowest_label_name)

    def read_csv(self, path):
        return pd.read_csv(path, sep='\t', encoding='mac_roman')

    def drop_unwanted(self, corpora):
        return corpora.drop(columns=['ID'])

    def divide_targets(self, target_name, corpora):
        target_name_df = pd.DataFrame(columns=['Target', 'Tweet', 'Stance'])
        target_name_df.Tweet = corpora.loc[corpora['Target'] == target_name, 'Tweet']
        target_name_df.Target = corpora.loc[corpora['Target'] == target_name, 'Target']
        target_name_df.Stance = corpora.loc[corpora['Target'] == target_name, 'Stance']
        print("\n"+target_name)
        return target_name_df

    def calculate_stance_labels(self, dataframe):
        lowest_label_name = ''
        lowest_size = 0
        against = 0
        favor = 0
        none = 0
        for stance in dataframe.Stance:
            if (stance == 'AGAINST'):
                against = against + 1
            elif (stance == 'FAVOR'):
                favor = favor + 1
            elif (stance == 'NONE'):
                none = none + 1
        if (against < favor and against < none):
            print("aganist is the least of the stance labels")
            lowest_label_name = "AGAINST"
            lowest_size = against
        elif (favor < against and favor < none):
            print("favor is the least of the stance labels")
            lowest_label_name = "FAVOR"
            lowest_size = favor
        elif (none < against and none < favor):
            print("favor is the least of the stance labels")
            lowest_label_name = "NONE"
            lowest_size = none
        if (against == favor):
            print("against and favor are equal")
        elif (against == none):
            print("against and none are equal")
        elif (none == favor):
            print("none and favor are equal")
        print("Against = "+str(against))
        print("Favor = "+str(favor))
        print("Favor = "+str(none))
        values_dic = dict();
        values_dic['lowest_size'] = lowest_size
        values_dic['lowest_label_name'] = lowest_label_name
        return values_dic

    def balance_labels(self, label, lowest_size, dataframe, target_name, lowest_label):
        target_label_balanced = []
        target_stance = []
        target_label_balanced_df = pd.DataFrame(columns=['Target', 'Tweet', 'Stance'])
        create_dataframe = pd.concat([dataframe])
        target_label = create_dataframe.loc[create_dataframe['Stance'] == label, 'Tweet']
        for i in range(lowest_size):
            target_stance.append(label)
        if (label != lowest_label):
            target_label_balanced = random.sample(target_label.to_list(), lowest_size)
            target_label_balanced_df['Tweet'] = target_label_balanced
        else:
            target_label_balanced_df['Tweet'] = target_label

        target_label_balanced_df['Stance'] = target_stance
        for i in range(lowest_size):
            target_label_balanced_df['Target'] = target_name
        return target_label_balanced_df

    def append_target_df(self, target_name):
        self.target_df= pd.concat([self.balance_labels('AGAINST', self.lowest_size, self.target, target_name, self.lowest_label_name)])
        self.target_df = pd.concat([self.target_df, self.balance_labels('NONE', self.lowest_size, self.target, target_name, self.lowest_label_name)])
        self.target_df = pd.concat([self.target_df, self.balance_labels('FAVOR', self.lowest_size, self.target, target_name, self.lowest_label_name)])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='F:/NLP/Project/Introduction_NLP/data/train_data_A.txt')
    parser.add_argument('--out_path', type=str, default='output/')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    target_names = ['Atheism', 'Climate Change is a Real Concern','Feminist Movement','Hillary Clinton','Legalization of Abortion']
    df = pd.DataFrame(columns=['ID', 'Target', 'Tweet', 'Stance'])
    for targets in target_names:
        balancer = balancingData(corpora_path=args.in_path, target_name=targets)
        df = pd.concat([df, balancer.target_df], axis=0)
    df.ID = range(1, df.shape[0]+1)
    df = df.sample(frac=1)
    df.to_csv("F:/NLP/Project/Introduction_NLP/data/balanced_data.txt", sep='\t', encoding='utf-8', index=None)
    print("saved...")