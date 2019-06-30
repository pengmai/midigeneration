from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2

import magenta
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

def get_pipeline(eval_ratio=0.1):
    partitioner = pipelines_common.RandomPartition(
        music_pb2.NoteSequence,
        ['eval_performances', 'train_performances'],
        [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    for mode in ['eval', 'train']:
        sustain_pipeline = note_sequence_pipelines.SustainPipeline(
            'SustainPipeline_' + mode)
        quantizer = note_sequence_pipelines.Quantizer(
            steps_per_second=STEPS_PER_SECOND, name='Quantizer_' + mode)
        perf_extractor = PerformanceExtractor(
            'PerformanceExtractor_' + mode,
            num_velocity_bins=NUM_VELOCITY_BINS)
        encoder_pipeline = OneHotEncoderPipeline(
            'EncoderPipeline_' + mode,
            num_velocity_bins=NUM_VELOCITY_BINS)

        dag[sustain_pipeline] = partitioner[mode + '_performances']
        dag[quantizer] = sustain_pipeline
        dag[perf_extractor] = quantizer
        dag[encoder_pipeline] = perf_extractor
        dag[dag_pipeline.DagOutput(mode + '_performances')] = encoder_pipeline

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

    values = [int(feat.float_list.value[0]) for feat in seq.feature_lists.feature_list['inputs'].feature]
    decoder = magenta.music.PerformanceOneHotEncoding(NUM_VELOCITY_BINS)
    performance = magenta.music.Performance(
        steps_per_second=120,
        num_velocity_bins=NUM_VELOCITY_BINS)
    for val in values:
        performance.append(decoder.decode_event(val))
    note_sequence = performance.to_sequence()
    magenta.music.sequence_proto_to_midi_file(note_sequence, outfile)

if __name__ == '__main__':
    processed = pipeline.load_pipeline(
        get_pipeline(),
        pipeline.tf_record_iterator('take2.tfrecord', music_pb2.NoteSequence))
    seq = processed['train_performances'][0]
    sequence_example_to_midi_file(seq, 'magenta-out.mid')