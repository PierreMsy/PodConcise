from typing import Tuple
import numpy as np


def softmax(logits: Tuple[np.ndarray]) -> Tuple[np.ndarray]: 
    
    e_logits = np.exp(logits)
    sum_e_logits = np.sum(e_logits, axis=1).reshape(-1,1)

    softmax = e_logits / sum_e_logits

    return softmax