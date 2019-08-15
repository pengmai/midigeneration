import argparse
import midi_io
import os

from data import Piece
from models import Markov, HiddenMarkov, NoteState


def generate_score(midi_file):
    '''(str) -> NoneType
    Generate the score (.xml) for the auto-generated midi file.
    Please open using a MusicXMl reader such as Finale NotePad.
    '''
    import music21   # required to display score
    midi_file_output = music21.converter.parse(midi_file)
    midi_file_output.show()


def collect_all_midis(directory):
    """Return a list of filenames of all midi files inside directory."""
    pieces = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.mid', '.midi']:
                pieces.append(os.path.join(dirpath, filename))
    return pieces


def main(args):
    if args.input.endswith('.mid') or args.input.endswith('.midi'):
        pieces = [args.input]
    else:
        pieces = collect_all_midis(args.input)
    print(f'Generating from {pieces} with context length {args.context or 1}')

    hmm = HiddenMarkov(chain_length=args.context or 1)
    mm = Markov(chain_length=args.context or 1)

    for piece in pieces:
        musicpiece = Piece(piece)
        # here all_bars is the list of chord labels generated for each bar
        key_sig, state_chain, all_bars, observations = NoteState.piece_to_state_chain(musicpiece)
        hmm.add(all_bars)
        hmm.train_obs(observations, key_sig)
        mm.add(state_chain)

    print('Training complete.')

    hidden_chain = hmm.generate_hidden()
    if len(hidden_chain) < 100:
        print(f'result too short: {len(hidden_chain)}')

    note_chain = hmm.generate(hidden_chain, mm.markov)
    notes = NoteState.state_chain_to_notes(note_chain, musicpiece.bar)
    song = [musicpiece.meta] + [[n.note_event() for n in notes]]
    print(f'bar number: {hmm.temporary_bar_counter}')
    print(f'klist succeeded: {hmm.temporary_klist_counter} times')
    print(f'regen_success: {hmm.regen_success_counter}')
    midi_io.write(args.output or 'output/output_hmm.mid', song)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='The name of the midi file to use')
    parser.add_argument('-s', '--start', type=int, help='The bar to start from')
    parser.add_argument('-e', '--end', type=int, help='The bar to end at')
    parser.add_argument('-o', '--output', help='The name of the generated file')
    parser.add_argument('-c', '--context', default=1, type=int, help='The context length used in the Markov models')

    main(parser.parse_args())
