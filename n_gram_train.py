"""Trains a new n-gram model."""

import argparse
import pickle

import numpy as np
from models.n_gram import NGramModel

def main(args):
    """Trains and saves a new n-gram model."""
    model = NGramModel(args.context, vocab_size=240)

    data = np.load(args.input, allow_pickle=True)
    model.train(data)

    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
    print('Training complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='The name of the npy file to use')
    parser.add_argument('-o', '--output', required=True, help='The location to save the model after training completes')
    parser.add_argument('-c', '--context', default=2, type=int, help='The context length used in the n-gram models')
    args = parser.parse_args()

    main(args)
