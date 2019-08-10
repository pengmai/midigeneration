"""
Script to convert either a single MIDI file or a tfrecord file consisting of
multiple MIDI files into numpy arrays.
"""

import argparse
import os

from preprocess.create_dataset import get_pipeline, encoded_vals_to_midi_file
from preprocess.truncate import sample_random
from magenta.scripts.convert_dir_to_note_sequences import convert_midi
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2

import numpy as np

def main(args):
    if args.input.endswith('.mid') or args.input.endswith('.midi'):
        sequence = convert_midi('', '', os.path.expanduser(args.input))
        pieces = [sequence]
    else:
        assert args.input.endswith('.tfrecord')
        pieces = pipeline.tf_record_iterator(args.input, music_pb2.NoteSequence)

    pipe = get_pipeline()
    processed = pipeline.load_pipeline(
        pipe,
        pieces)

    performance_values = []
    for i, perf in enumerate(processed['train_performances']):
        values = [int(feat.float_list.value[0]) for feat in perf.feature_lists.feature_list['inputs'].feature]
        values = np.array(values)
        performance_values.append(values)
    truncated = sample_random(performance_values, max_len=1024)
    np.save(args.output, truncated)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='The name of the tfrecord file to read from')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='The name of the .npy file')
    args = parser.parse_args()

    main(args)
