"""Functions to handle data arrays

"""

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def round_nonzero_decimal(num, precision=1, method="round"):
    """
    Round an input decimal number starting from its first non-zero value.

    For instance, with precision of 1, we have:
    0.09324 -> 0.09
    0.00246 -> 0.002

    Parameters
    ----------
    num : float
        Float number.
    precision : int
        Number of decimals to keep. Defaults to 1.
    method : str
        Method for rounding a number. Currently supports
        np.round(), np.floor(), and np.ceil().

    Returns
    -------
    round_num : float
        Rounded number.
    """
    # Validation
    if num > 1:
        raise ValueError("num should be less than 1.")
    if num == 0: return 0
    
    # Identify the number of zero decimals
    decimals = int(abs(np.floor(np.log10(abs(num)))))
    precision = decimals + precision - 1
    
    # Round decimal number
    if method == "round":
        round_num = np.round(num, precision)
    elif method == "floor":
        round_num = np.true_divide(np.floor(num * 10 ** precision), 10 ** precision)
    elif method == "ceil":
        round_num = np.true_divide(np.ceil(num * 10 ** precision), 10 ** precision)
    
    return round_num

def round_up_half(num, decimals=0):
    """
    Round a number using a 'round half up' rule. This function always
    round up the half-way values of a number.

    NOTE: This function is added because Python's default round() 
    and NumPy's np.round() functions use 'round half to even' method.
    Their implementations mitigate positive/negative bias and bias 
    toward/away from zero, while this function does not. Hence, this 
    function should be preferentially only used for the visualization 
    purposes.

    Parameters
    ----------
    num : float
        Float number.
    decimals : int
        Number of decimals to keep. Defaults to 0.

    Returns
    -------
    round_num : float
        Rounded number.
    """
    multiplier = 10 ** decimals
    round_num = float(
        Decimal(num * multiplier).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / multiplier
    )
    return round_num