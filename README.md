# Introduction_NLP
Stance Classification in Tweets

## Do Pre-Processing
  ##### Step 1: Execute `python3` `preprocessor.py` `--in_path` `--out_path` `--regex` `tokenizer` <br />
  `in_path` is the data to be preprocessed. Default file is `data/semeval2016-task6-trainingdata.txt` <br />
  `out_path` should be the location of your output data. Default location is `output/` <br />
  ##### Step 2: Pass in hyperparameters for further tunning: <br />
  `regex` accepts the pattern you wish to format the `in_path` file with. Default is `[^A-Za-z]+` <br />
  `tokenizer` accepts the format for tokenization. Default is `r'\w+'`
