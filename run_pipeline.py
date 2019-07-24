"""Script to convert a tfrecord file into numpy arrays."""

import argparse

from preprocess.create_dataset import get_pipeline
from magenta.pipelines import pipeline
from magenta.protobuf import music_pb2

import numpy as np

def main(args):
    pipe = get_pipeline()
    all_pieces = list(pipeline.tf_record_iterator(args.input, music_pb2.NoteSequence))
    # TODO: Remove this. Just process the maestro set 10 pieces at a time to
    # avoid overloading memory
    pieces = all_pieces[80:90]
    processed = pipeline.load_pipeline(
        pipe,
        pieces)

    performance_values = []
    for perf in processed['train_performances']:
        values = [int(feat.float_list.value[0]) for feat in perf.feature_lists.feature_list['inputs'].feature]
        values = np.array(values)
        performance_values.append(values)
    np.save(args.output, performance_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='The name of the tfrecord file to use')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='The name of the .npy file')
    args = parser.parse_args()

    main(args)
