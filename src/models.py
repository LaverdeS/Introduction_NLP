import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle

class Logisticregression:
    def __init__(self, tfidf_path, stance_path, multiclass, penalty, solver, c_value, max_itr, filename):
        print("LG constructor...")
        self.tfidf = self.read_tfidf(tfidf_path)
        self.stance = self.read_stance(stance_path)
        self.x_train, self.x_test, self.y_train, self.y_test = self.test_train_split(self.tfidf,self.stance);
        self.logmodel = self.logistic_regression(multiclass, penalty, solver, c_value, max_itr)
        self.logmodel.fit(self.x_train,self.y_train)
        self.training_accuracy = self.accuracy_training(self.x_train, self.y_train)
        self.y_pred = self.logmodel.predict(self.x_test)
        self.testing_accuracy = self.accuracy_testing(self.x_test, self.y_test)
        self.cm = self.confusion_matrix(self.y_test, self.y_pred)
        self.p_r_f = self.precison_recall_f1(self.y_test, self.y_pred)
        self.test_train_split(self.tfidf, self.stance)
        self.save(filename, self.logmodel)


    def read_tfidf(self, path):
        return pd.read_csv(path, sep="\t")


    def read_stance(self, path):
        stance_df = pd.read_csv(path, sep="\t", encoding="utf-8")
        # print(stance_df.columns)
        stance_df = stance_df.drop(columns=['Unnamed: 0'])
        # print(stance_df.columns)
        return stance_df


    def test_train_split(self, tfidf, stance):
        return train_test_split(tfidf, stance)


    def logistic_regression(self, multiclass, penalty, solver, c_value, max_itr):
        return LogisticRegression(multi_class=multiclass, penalty=penalty, solver=solver, C=c_value, max_iter=max_itr)


    def accuracy_training(self, x_train, y_train):
        return metrics.accuracy_score(y_train, self.logmodel.predict(x_train))


    def accuracy_testing(self, x_test, y_test):
        return metrics.accuracy_score(y_test, self.logmodel.predict(x_test))


    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)


    def precison_recall_f1(self, y_test, y_pred):
        return metrics.classification_report(y_test, y_pred, digits=3)


    def save(self, filename, logmodel):
        with open(filename, 'wb') as file:
            pickle.dump(logmodel, file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="lg")
    parser.add_argument('--tfidf_file', type=str, default="./hillary_tfidf.tsv")
    parser.add_argument('--stance', type=str, default="./stance_hillary.csv")
    parser.add_argument('--multiclass', type=str, default='multinomial')
    parser.add_argument('--penalty', type=str, default='none')
    parser.add_argument('--solver', type=str, default='lbfgs')
    parser.add_argument('--c', type=float, default=0.01)
    parser.add_argument('--max_itr', type=int, default=200)
    parser.add_argument('--filename', type=str, default='atheism_LR_1.sav')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.model == "lg":
        seeds = 0;
        if args.tfidf_file == "F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv":
            seeds = 1000000
        elif args.tfidf_file == "F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv":
            seeds = 1
        elif args.tfidf_file == "F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv":
            seeds = 5555
        elif args.tfidf_file == "F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv":
            seeds = 987654321
        elif args.tfidf_file == "F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv":
            seeds = 987654321
        #seeds = [1, 49, 999, 5555, 54321, 987654, 1000000, 12345678, 987654321, 1234567890]
        np.random.seed(seeds)
        model = Logisticregression(tfidf_path=args.tfidf_file, stance_path=args.stance, multiclass=args.multiclass,
                                       penalty=args.penalty, solver=args.solver, c_value=args.c, max_itr=args.max_itr,
                                     filename=args.filename)
        # print(model.training_accuracy)
        print(model.testing_accuracy)
        print(seeds)
        print(model.cm)
        print(model.p_r_f)

    else:
        print("not implemented...", flush=True)
        # model =
