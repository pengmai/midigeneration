"""Generates from a trained n-gram model."""

import argparse
import pickle
from models.n_gram import NGramModel
from preprocess.create_dataset import encoded_vals_to_midi_file

def main(args):
    with open(args.input, 'rb') as f:
        model = pickle.load(f)
    output = model.generate(args.steps, verbose=True)
    encoded_vals_to_midi_file(output, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='The location of the trained n-gram model')
    parser.add_argument('-o', '--output', required=True, help='The location to save the generated midi file')
    parser.add_argument('-s', '--steps', default=500, type=int, help='The number of steps to generate for')
    main(parser.parse_args())
