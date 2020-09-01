import pandas as pd
import argparse


def walk_here(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for dir in dirnames:
            walk_here(dirpath + "/" + dir)
        files.extend(filenames)
    return files


def evaluate_predictions(predictions_txt, truth_csv):
    with open(predictions_txt, 'r', encoding='utf-8') as file_in:
        preds = file_in.readlines()
        predicts = []
        for pred in preds:
            pred = pred[0:10].replace("'", "").replace("[", '').replace("(", '').replace(" ", '').replace(",", '')
            predicts.append(pred)

    df_true = pd.read_csv(truth_csv, encoding='utf-8', header=None)

    #print(f'df_true.head: \n{df_true.head(3)}')
    df_true.drop(index=0, inplace=True)
    true = df_true.iloc[:, 0]
    #print(f'df_true.head: \n{df_true.head(3)}')

    correct = 0
    wrong = 0

    for prediction, truth in zip(predicts, true[0:len(predicts)+1]):
        prediction = prediction.replace("'", "").replace("[", '').replace("(", '')
        print(prediction, truth)
        if prediction == truth:
            correct += 1
            pass
        wrong += 1

    print(predictions_txt)
    print(f'correct: {correct}, wrong: {wrong}, accuracy= {round(correct / (correct + wrong) * 100)}%')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default='')
    parser.add_argument('--truth', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if args.pred[-3:] == 'txt':
        walked_files = walk_here(args.dataset_filepath)
        for pred_file in walked_files:
            evaluate_predictions(pred_file, args.truth)
    else:
        evaluate_predictions(args.pred, args.truth)

