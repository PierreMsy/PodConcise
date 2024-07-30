import hashlib
import numpy as np
import pandas as pd
from time import sleep
from typing import Sequence, Tuple, Union, Dict
from itertools import product
from html import unescape
from unidecode import unidecode

import translators as ts
from datasets import Dataset, load_metric
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sklearn.metrics import log_loss

from podconcise.utils import softmax
from podconcise.constant import DATA_SCIENCE_GUESTS



def preprocess_podcasts(df_podcast: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for NLP treatment on the podcast DataFrame.
    - Remove non standard symbol and lower guest and title.
    - Add unique and deterministic id.
    """
    cols_to_pp = ["guest", "title"]
    for col in cols_to_pp:
        df_podcast[col] = np.vectorize(lambda title: unidecode(title))(df_podcast[col])
        df_podcast[col] = df_podcast[col].str.lower()

    hash_str_vec = np.vectorize(hash_str)
    df_podcast.insert(0, 'id', hash_str_vec(df_podcast.guest + df_podcast.title))

    assert df_podcast.id.nunique() == len(df_podcast), "colision in the hashing to create unique id!"

    return df_podcast


def hash_str(
        s:str,
        nbr_digits:int = 8,
        encoding:str = "utf-8"
    ) -> int:
    """
    Get a int resulting from the hashing of a str.
    """
    hash = int(hashlib.sha1(s.encode(encoding)).hexdigest(), 16) % (10 ** nbr_digits)
    
    # if the int ends with 0, add random but seeded digits to pad at the begginning
    # in order to always have the asked nbr of digits when represented as str.
    if len(str(hash)) < nbr_digits:
        
        missing_digits = nbr_digits - len(str(hash))
        np.random.seed(hash) 
        new_digits = np.random.randint(10**(missing_digits-1), 10**missing_digits)
        hash = hash * 10 ** missing_digits + new_digits
        
    return hash


def back_translate(
    txt: str,
    translator: str,
    intermediate_language: str, 
    original_language: str="en",
    sleep_between_calls: float=1.5
    ) -> str:
    """
    Create a most likely new text from translating the given text in the given intermediate language
    using the given translator and translating back to the original language.
    """
    sleep(sleep_between_calls)
    txt_translated = None
    try:
        txt_translated = ts.translate_text(
            txt,
            translator=translator,
            from_language=original_language,
            to_language=intermediate_language,
        )
    except Exception as e:
        translation_error= (
            f"Unable to translate from {original_language} to {intermediate_language} " +
            f"using {translator} the text '{txt}'")
        print(translation_error)
        print(e)


    txt_back_translated = None
    if txt_translated:
        try:
            txt_back_translated = ts.translate_text(
                txt_translated,
                translator=translator,
                from_language=intermediate_language,
                to_language=original_language,
            )
        except Exception as e:
            translation_error= (
                f"Unable to translate from {intermediate_language} to {original_language} " + 
                f"using {translator} the text '{txt_translated}'")
            print(translation_error)
            print(e)

    return txt_back_translated


def augment_with_backtranslation(
    txt:str,
    translators: Sequence[str],
    intermediate_languages: Sequence[str],
    return_original_text: bool=False
    )-> Union[Sequence[str], Tuple[str, Sequence[str]]]:
    """
    Agument the given str by back translating to and from all given intermediate languages
    using all the given product (cross product between the two).
    Return the created str in lower case.
    Following argument can return only the newly generated text or return them alongside the original text.
    """
    try:
        new_txts = []
        for translator, language in product(translators, intermediate_languages):
            new_txt = back_translate(txt, translator, language)
            if not new_txt:
                # pass on translation error
                continue
            new_txts.append(unescape(new_txt).lower())

        new_txts_deduplicated = [new_txt for new_txt in set(new_txts) if new_txt != txt]

        if return_original_text:
            res = (txt, new_txts_deduplicated)
        else:
            res = new_txts_deduplicated
            
    except Exception as e:
        print(txt)
        raise(e)

    return res


def tokenize_title(batch: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dict[str, np.ndarray]:
    """
    Tokenize a given batch of titles.
    """
    batch_encoded = tokenizer(
        batch['title'],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=256,
    )
    return batch_encoded


metric = load_metric("glue", "mrpc", trust_remote_code=True) # f1 & accuracy

def compute_classification_metrics(eval_preds: Tuple[np.ndarray[np.ndarray[float]], np.ndarray[int]]) -> Dict[str, float]:
    """
    Compute a metric dictionary from fiven logit predictions and true labels.
    """
    logits, labels = eval_preds
    pred_labels = np.argmax(logits, axis=-1)

    proba = softmax(logits)
    proba_1 = proba[:,1]
    log_loss_value = log_loss(labels, proba_1)

    metrics = metric.compute(predictions=pred_labels, references=labels)
    metrics["log_likelihood"] = - log_loss_value

    metrics["f1_plus_log_likelihood"] = metrics["f1"] - log_loss_value

    return metrics