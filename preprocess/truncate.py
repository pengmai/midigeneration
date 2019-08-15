"""Functions to handle truncating datasets"""

import numpy as np

def sample_random(dataset, max_len=1024, start_token=375, verbose=True):
    """
    Samples a chunk of max_len for each piece in the dataset.
    """
    untouched = 0
    truncated = []
    for piece in dataset:
        if len(piece) <= max_len:
            untouched += 1
            if start_token is not None:
                piece = np.insert(piece, 0, start_token)
            truncated.append(piece)
        else:
            start = np.random.randint(0, len(piece) - max_len)
            piece = piece[start:start + max_len]
            if start_token is not None:
                piece[0] = start_token
            truncated.append(piece)
    if verbose:
        print((f'Skipped truncating {untouched}/{len(dataset)} pieces '
               f'for being shorter than {max_len} tokens long'))
    return truncated
