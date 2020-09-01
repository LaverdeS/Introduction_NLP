import os
import torch
import logging
import pandas as pd
import argparse
import os
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--train_file', type=str, default='')
    parser.add_argument('--val_file', type=str, default='')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument('--labels_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    home = os.getcwd()
    output_dir = home + '/output/' + args.output_dir
    os.mkdir('./output/' + args.output_dir)

    print("start...")
    os.environ["HDF5_DISABLE_VERSION_CHECK"] = '1'

    DATA_PATH = args.data_path
    LABEL_PATH = args.label_path

    databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                              tokenizer='bert-base-uncased',
                              train_file=args.train_file,
                              val_file=args.val_file,
                              label_file=args.labels_file,
                              text_col='text',
                              label_col='label',
                              batch_size_per_gpu=4,
                              max_seq_length=512,
                              multi_gpu=True,  # Here I might choose not to use multiple ones if unexiting and error...
                              multi_label=False,  # Set to True for multilabel
                              model_type='bert')

    print("\nLet's build the learner...\n")

    logger = logging.getLogger()
    device_cuda = torch.device("cuda") # device cuda here for gammaweb
    metrics = [{'name': 'accuracy', 'function': accuracy}]

    OUTPUT_DIR = output_dir

    learner = BertLearner.from_pretrained_model(
        databunch,
        pretrained_path='bert-base-uncased',
        metrics=metrics,
        device=device_cuda, # check variable, could be going to cpu
        logger=logger,
        output_dir=OUTPUT_DIR,
        finetuned_wgts_path=None,
        warmup_steps=5, # Originally in 500... 50 for test purposes
        multi_gpu=True,
        is_fp16=True,
        multi_label=False, # Set here to True for multi-label
        logging_steps=50) # number of steps between each tensorboard metrics calculation. Set it to 0 to disable tensor flow logging. Keeping this value too low will lower the training speed as model will be evaluated each time the metrics are logged

    learner.fit(epochs=3, # Originally on 6
                lr=1e-4,
                validate=True, 	# Evaluate the model after each epoch
                schedule_type="warmup_cosine",
                optimizer_type="lamb") # 'lamb' here for speed up

    learner.save_model()
    print("done...")