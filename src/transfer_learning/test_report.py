import pandas as pd
from fast_bert.prediction import BertClassificationPredictor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, \
    jaccard_score  # ,jaccard_similarity_score is accuracy_score when multiclass
import argparse
import os

os.environ["HDF5_DISABLE_VERSION_CHECK"] = '2'


def read_predictions(predictions_file):
    with open(predictions_file, 'r', encoding='utf-8') as file_in:
        preds = file_in.readlines()
        predicts = []
        for pred in preds:
            pred = pred[0:10].replace("'", "").replace("[", '').replace("(", '').replace(" ", '').replace(",", '')
            predicts.append(pred)
    return predicts


def generate_metrics_report(df_test, df_true, name):
    df_true_ = pd.read_csv(df_true, encoding='utf-8', header=None)
    texts = df_test.text.tolist()
    multiple_predictions = predictor.predict_batch(texts)
    path = name + args.file_out_tag

    with open('./predictions/' + path, 'w') as filehandle:
        for listitem in multiple_predictions:
            filehandle.write('%s\n' % listitem)

    df_true_.drop(index=0, inplace=True)
    true = df_true_.iloc[:, 0]

    evaluate_predictions('./predictions/' + path, true, name.replace('.csv', ''))

    print(f'\nprecision_recall_fscore_support:')
    prf1s(true, './predictions/' + path)


def prf1s(truth, predictions):
    predicts = read_predictions(predictions)
    extended = precision_recall_fscore_support(truth, predicts, average=None, labels=["FAVOR", "AGAINST", "NONE"])
    averaged = precision_recall_fscore_support(truth, predicts, average="macro", labels=["FAVOR", "AGAINST", "NONE"])
    averaged_2 = precision_recall_fscore_support(truth, predicts, average="macro", labels=["FAVOR", "AGAINST"])
    micro = precision_recall_fscore_support(truth, predicts, average="micro", labels=["FAVOR", "AGAINST", "NONE"])
    # micro_2 = precision_recall_fscore_support(truth, predicts, average="micro", labels=["FAVOR", "AGAINST"])
    metric_names = ['precission', 'recall', 'f1', 'support']

    for ix, label in zip(range(0, 3), ["FAVOR", "AGAINST", "NONE"]):
        print(f'\nlabel: {label}')
        report_label = draft_report[label]
        for metric, metric_ext in zip(metric_names, extended):
            print(f'{metric}: {metric_ext[ix]}')
            if metric != 'support':
                draft_report[label][metric] = metric_ext[ix]


    print(f'\nlabel: AVERAGE("macro")')
    for metric, metric_avg in zip(metric_names, averaged):
        print(f'{metric}: {metric_avg}')
        if metric != 'support':
            average_ = draft_report['AVERAGE']
            average_[metric] = metric_avg

    print(f'\nlabel: AVERAGE_2("macro")')
    for metric, metric_avg in zip(metric_names, averaged_2):
        print(f'{metric}: {metric_avg}')
        if metric != 'support':
            average_ = draft_report['AVERAGE_2']
            average_[metric] = metric_avg

    print(f'\nlabel: AVERAGE("micro")')
    for metric, metric_avg in zip(metric_names, micro):
        print(f'{metric}: {metric_avg}')
        if metric != 'support':
            average_ = draft_report['micro']
            average_[metric] = metric_avg

    #print(f'\nlabel: AVERAGE_2("micro")')
    #for metric, metric_avg in zip(metric_names, micro_2):
    #    print(f'{metric}: {metric_avg}')
    #    if metric != 'support':
    #        average_ = draft_report['micro']
    #        average_[metric] = metric_avg


def evaluate_predictions(predictions_txt, true, name_):
    print(f"\n--- METRICS REPORT for {name_} ---")
    predicts = read_predictions(predictions_txt)
    correct = 0

    for prediction, truth in zip(predicts, true):
        prediction = prediction.replace("'", "").replace("[", '').replace("(", '')
        if prediction == truth:
            correct += 1
            pass
    wrong = len(predicts) - correct
    oa = correct / (correct + wrong)
    ascore = accuracy_score(true, predicts)
    jaccard = jaccard_score(true, predicts, average="macro")
    blacc = balanced_accuracy_score(true, predicts)
    print(f'correct: {correct}, wrong: {wrong}, \noverall_accuracy : {oa}')
    # print(f'sklearn accuracy_score correct samples: {accuracy_score(true, predicts, normalize=False)}')
    print(f'sklearn accuracy_score (average accuracy): {ascore}')
    # print(f'sklearn jaccard_similarity_score: {jaccard_similarity_score(true, predicts)}')
    print(f'sklearn jaccard_score("macro"): {jaccard}')
    print(f'sklearn balanced_accuracy_score: {blacc}')

    draft_report['correct'] = correct
    draft_report['wrong'] = wrong
    draft_report['overall_accuracy'] = oa
    draft_report['average_accuracy'] = ascore
    draft_report['jaccard_macro'] = jaccard
    draft_report['balanced_accuracy'] = blacc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str, default='./data_split/hillary_test/')
    parser.add_argument('--model_path', type=str, default='hillary_10_e-3')
    parser.add_argument('--label_path', type=str, default='./')
    parser.add_argument('--file_out_tag', type=str, default='_report.txt')
    parser.add_argument('--truth', type=str, default='./data_split/hillary_truth/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    print(f'arguments: {args}')
    os.environ["HDF5_DISABLE_VERSION_CHECK"] = '2'
    OUTPUT_DIR = './output/'
    MODEL_PATH = OUTPUT_DIR + args.model_path + '/model_out'
    LABEL_PATH = args.label_path

    categ = args.model_path[0:8].replace('_','')
    print(f'running for category: {categ}!')

    draft_report = {
	'name':args.model_path,
        'correct': 0,
        'wrong': 0,
        'overall_accuracy': 0.0,
        'average_accuracy': 0.0,
        'jaccard_macro': 0.0,
        'balanced_accuracy': 0.0,
        'FAVOR': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0},
        'AGAINST': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0},
        'NONE': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0},
        'AVERAGE': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0},
        'AVERAGE_2': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0},
        'micro': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0}
        #'micro_2': {'precission': 0.0, 'recall': 0.0, 'f1': 0.0}
    }

    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=LABEL_PATH,
        multi_label=False,
        model_type='bert',
        do_lower_case=False)

    if args.file_in[-3:] == 'csv':
        df_in = pd.read_csv(args.file_in, encoding='utf-8')
        truth_file =  args.truth
        generate_metrics_report(df_in, truth_file, name=args.model_path)
    else:
        *_, truth_files =  list(next(os.walk(args.truth)))
        *_, test_files = list(next(os.walk(args.file_in)))
        print(f'lenght of truth_files: {len(truth_files)}')
        print(f'lenght of test_files: {len(test_files)}')
        print(f'tests: {test_files}, \ntruth: {truth_files}')        

        TEST_REPORT = pd.DataFrame()

        for true_one, tested_one in zip(truth_files, test_files):
            df_in = pd.read_csv(args.file_in + tested_one, encoding='utf-8')
            generate_metrics_report(df_in, args.truth + true_one, name=args.model_path)

            draft_report['FAVOR_p'] = draft_report['FAVOR']['precission']
            draft_report['FAVOR_r'] = draft_report['FAVOR']['recall']
            draft_report['FAVOR_f1'] = draft_report['FAVOR']['f1']

            draft_report['AGAINST_p'] = draft_report['AGAINST']['precission']
            draft_report['AGAINST_r'] = draft_report['AGAINST']['recall']
            draft_report['AGAINST_f1'] = draft_report['AGAINST']['f1']

            draft_report['NONE_p'] = draft_report['NONE']['precission']
            draft_report['NONE_r'] = draft_report['NONE']['recall']
            draft_report['NONE_f1'] = draft_report['NONE']['f1']

            draft_report['macro_p'] = draft_report['AVERAGE']['precission']
            draft_report['macro_r'] = draft_report['AVERAGE']['recall']
            draft_report['macro_f1'] = draft_report['AVERAGE']['f1']

            draft_report['macro2_p'] = draft_report['AVERAGE_2']['precission']
            draft_report['macro2_r'] = draft_report['AVERAGE_2']['recall']
            draft_report['macro2_f1'] = draft_report['AVERAGE_2']['f1']

            draft_report['micro_p'] = draft_report['micro']['precission']
            draft_report['micro_r'] = draft_report['micro']['recall']
            draft_report['micro_f1'] = draft_report['micro']['f1']

            #draft_report['micro2_p'] = draft_report['micro_2']['precission']
            #draft_report['micro2_r'] = draft_report['micro_2']['recall']
            #draft_report['micro2_f1'] = draft_report['micro_2']['f1']

            final_report = draft_report.copy()
            del final_report['FAVOR']
            del final_report['AGAINST']
            del final_report['NONE']
            del final_report['AVERAGE']
            print(f'\nfinal_report: {final_report}')
            TEST_REPORT = TEST_REPORT.append(final_report, ignore_index=True)
            print('\nreport appended!\n')
        TEST_REPORT = TEST_REPORT.reindex(columns=['name','correct','wrong','overall_accuracy','average_accuracy','balanced_accuracy','jaccard_macro','FAVOR_p','FAVOR_r','FAVOR_f1','AGAINST_p','AGAINST_r','AGAINST_f1','NONE_p','NONE_r','NONE_f1','micro_p','micro_r','micro_f1','macro2_p','macro2_r','macro2_f1','macro_p','macro_r','macro_f1'])
        TEST_REPORT.to_csv('./OVERALL_REPORT_'+categ+'.csv', encoding='utf-8', mode='a', header=False, index=None)
