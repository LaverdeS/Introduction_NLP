# Introduction_NLP
Stance Classification in Tweets

## Pre-Processing
  ##### Step 1: Execute `python3` `preprocessor.py` `--in_path` `--out_path` `--remove_numbers` `--remove_special_characters` `--remove_stopwords` `--stem` <br />
  `in_path` is the data to be preprocessed. Default file is `data/semeval2016-task6-trainingdata.txt` <br />
  `out_path` should be the location of your output data. Default location is `output/` <br />
  ##### Step 2: Pass in hyperparameters for further tunning: <br />
   `remove_numbers` will remove all digits from 0 to 9 <br />
  `remove_special_characters` will remove sepecial characters from the dataset<br />
  `remove_stopwords` will remove English sopwords <br />
  `stem` will apply stemming on the dataset <br />
  
## Feature Engineering
### Calculation of Term Frequency - Inverse Document Frequency `TF-IDF` was done using the following procedure:
##### `fit_transformer()` <br />
##### `get_feature_names()` is made `index` of the `DataFrame` <br />
##### `todense()` is applied to make the Dataframe dense <br />
##### `transpose()` replaces row with columns and columns with rows to have the Bag of Words (BOW) on as `columns` instead of `rows`  <br />

## RESTful-API
#### To read the documentation and the format of the POST requests, run restapi.py and from URL go to /docs and/or /redoc
