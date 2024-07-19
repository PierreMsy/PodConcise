import hashlib
import numpy as np


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