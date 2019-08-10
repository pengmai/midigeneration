from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2

import itertools
import magenta
import numpy as np
import tensorflow as tf

STEPS_PER_SECOND = 100 # TODO: Figure out how to properly set this.
NUM_VELOCITY_BINS = 32

class OneHotEncoderPipeline(pipeline.Pipeline):
    """
    A Pipeline that converts performances into the one hot encoding used by the
    magenta performance_rnn model.
    """
    def __init__(self, name, num_velocity_bins=0):
        super(OneHotEncoderPipeline, self).__init__(
            input_type=magenta.music.performance_lib.BasePerformance,
            output_type=tf.train.SequenceExample,
            name=name)
        self.encoder_decoder = magenta.music.OneHotIndexEventSequenceEncoderDecoder(
            magenta.music.PerformanceOneHotEncoding(
                num_velocity_bins=num_velocity_bins))

    def transform(self, performance):
        return [self.encoder_decoder.encode(performance)]

class PerformanceExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks from a quantized NoteSequence."""
    def __init__(self, name, num_velocity_bins=0,
                 note_performance=False):
        super(PerformanceExtractor, self).__init__(
            input_type=music_pb2.NoteSequence,
            output_type=magenta.music.performance_lib.BasePerformance,
            name=name)
        self._num_velocity_bins = num_velocity_bins
        self._note_performance = note_performance

    def transform(self, quantized_sequence):
        performances, stats = magenta.music.extract_performances(
            quantized_sequence,
            num_velocity_bins=self._num_velocity_bins,
            note_performance=self._note_performance)
        self._set_stats(stats)
        return performances

def pipeline_list_to_dict(lst):
    def pairwise(iterable):
        """Returns an iterator over every pair of elements in iterable."""
        first, second = itertools.tee(iterable)
        next(second, None)
        return zip(first, second)

    dag = {}
    for first, second in pairwise(lst):
        dag[second] = first
    return dag

def get_pipeline(eval_ratio=0.0):
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    # stretch_factors = [0.95, 0.975, 1, 1.025, 1.05]
    stretch_factors = [0.975, 1, 1.025]

    # Transpose up and down a major third.
    transposition_range = range(-3, 4)

    partitioner = pipelines_common.RandomPartition(
        music_pb2.NoteSequence,
        ['eval_performances', 'train_performances'],
        [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    for mode in ['eval', 'train']:
        sustain_pipeline = note_sequence_pipelines.SustainPipeline(
            'SustainPipeline_' + mode)
        stretch_pipeline = note_sequence_pipelines.StretchPipeline(
            stretch_factors,
            name='StretchPipeline_' + mode)
        splitter = note_sequence_pipelines.Splitter(
            hop_size_seconds=30.0,
            name='Splitter_' + mode)
        quantizer = note_sequence_pipelines.Quantizer(
            steps_per_second=STEPS_PER_SECOND, name='Quantizer_' + mode)
        transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
            transposition_range, name='TranspositionPipeline_' + mode)
        perf_extractor = PerformanceExtractor(
            'PerformanceExtractor_' + mode,
            num_velocity_bins=NUM_VELOCITY_BINS)
        encoder_pipeline = OneHotEncoderPipeline(
            'EncoderPipeline_' + mode,
            num_velocity_bins=NUM_VELOCITY_BINS)

        # These pipelines can be commented/uncommented to turn them on and off.
        pipelines = [
            sustain_pipeline,
            # stretch_pipeline,
            # splitter,
            quantizer,
            # transposition_pipeline,
            perf_extractor,
            encoder_pipeline,
            dag_pipeline.DagOutput(mode + '_performances')
        ]
        dag[pipelines[0]] = partitioner[mode + '_performances']
        dag.update(pipeline_list_to_dict(pipelines))

    return dag_pipeline.DAGPipeline(dag)

def sequence_example_to_midi_file(sequence, outfile='sequence-out.mid'):
    """
    Converts a SequenceExample representing a Performance into a MIDI file.

    Parameters
    ----------
    sequence : tf.train.SequenceExample
        The sequence representing the performance.
    outfile : str
        The name of the MIDI file to write to.
    """

    values = [int(feat.float_list.value[0]) for feat in sequence.feature_lists.feature_list['inputs'].feature]
    encoded_vals_to_midi_file(values, outfile=outfile)

def encoded_vals_to_midi_file(values, outfile='sequence-out.mid'):
    decoder = magenta.music.PerformanceOneHotEncoding(NUM_VELOCITY_BINS)
    performance = magenta.music.Performance(
        steps_per_second=100,
        num_velocity_bins=NUM_VELOCITY_BINS)
    for val in values:
        performance.append(decoder.decode_event(val))
    note_sequence = performance.to_sequence()
    magenta.music.sequence_proto_to_midi_file(note_sequence, outfile)

def run_pipeline(tfrecord_file):
    processed = pipeline.load_pipeline(
        get_pipeline(),
        pipeline.tf_record_iterator(tfrecord_file, music_pb2.NoteSequence))

    for i, perf in enumerate(processed['train_performances']):
        sequence_example_to_midi_file(perf, f'Undertale Dataset/performance-{i}.mid')
    return processed
