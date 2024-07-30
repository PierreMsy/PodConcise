"""
Given a model name corresponding to a model dumped in ../model and a 
dataset name corresponding to a pandas DataFrame dumped in ../data, this 
script will predict the title column and dump the resulting prediction DataFrame.
"""

import logging
import argparse
import os
import joblib
import math
from tqdm import tqdm
import numpy as np
import pandas as pd

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from transformers.tokenization_utils_base import (
    BatchEncoding, PreTrainedTokenizerBase
)
from datasets import Dataset
import torch

from podconcise.nlp_utils import tokenize_title


# *** settings & argument parsing.
PATH_DATA = r"../data"
PATH_MODEL = r"../model"
FILE_OUTPUT = r"df_classify_title_predicted"
BATCH_SIZE = 8

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    help='model name.'
)
parser.add_argument(
    '--dataset',
    type=str,
    help='Pandas dataset name.'
)
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset


# *** loading inputs.
path_dataset = os.path.join(PATH_DATA, dataset_name)
df_dataset = joblib.load(path_dataset)
dataset_test = Dataset.from_pandas(df_dataset)

path_model = os.path.join(PATH_MODEL, model_name)
model = DistilBertForSequenceClassification.from_pretrained(path_model, low_cpu_mem_usage=True)
tokenizer = DistilBertTokenizer.from_pretrained(path_model, low_cpu_mem_usage=True)


# *** inference.
total_preds = []
iter_dataset= dataset_test.iter(batch_size=BATCH_SIZE)

for batch in tqdm(iter_dataset, total=math.ceil(dataset_test.num_rows/BATCH_SIZE)):
    
    inputs = tokenize_title(batch, tokenizer)
    outputs = model(**inputs)
    logits = outputs.logits
    proba = torch.nn.functional.softmax(logits, dim=1)
    pred = torch.argmax(proba, dim=1).numpy()
    total_preds.append(pred)

predictions = np.concatenate(total_preds)
log_msg = (
    "Prediction done, repartition of predicted classes:\n" +
    f"{pd.Series(predictions).value_counts()}"
)
logging.info(log_msg)

df_dataset["pred"] = predictions
path_output = os.path.join(PATH_DATA, FILE_OUTPUT)
joblib.dump(df_dataset, path_output)
logging.info(f"Prediction file dumped at: {path_output}")



