import hashlib
import numpy as np
from time import sleep
from typing import Sequence, Tuple, Union


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