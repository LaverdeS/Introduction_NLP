import pandas as pd
from fast_bert.prediction import BertClassificationPredictor
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--file_out', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.file_in)


    OUTPUT_DIR = './output/'
    MODEL_PATH = OUTPUT_DIR + args.model_path+'/model_out'
    LABEL_PATH = './'

    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=LABEL_PATH,  # location for labels.csv file
        multi_label=False,
        model_type='bert',
        do_lower_case=False)

    df = pd.read_csv(LABEL_PATH + args.file_in, encoding='utf-8')
    texts = df.text.tolist()
    multiple_predictions = predictor.predict_batch(texts)
    # print(multiple_predictions)
    # print(type(multiple_predictions))

    with open('./predictions/'+args.file_out, 'w') as filehandle:
        for listitem in multiple_predictions:
            filehandle.write('%s\n' % listitem)
