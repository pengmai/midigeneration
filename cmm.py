# Markov Model thingy
import argparse
import random, glob
import data, midi, experiments, patterns, chords
from state import NoteState, SegmentState
import pickle
import copy

class Markov:

    '''
    Generic object for a Markov model

    trains state and state transitions by reading statechains
    statechain: a list of states
    state: a concrete class derived from the abstract class State

    '''
    START_TOKEN = 'start_token'
    STOP_TOKEN = 'stop_token'

    def __init__(self, chain_length=1):
        self.chain_length = chain_length
        self.markov = {}
        self.states = set()
        self.state_chains = [[]]

    def add(self, chain):
        '''
        add a statechain to the markov model (i.e. perform training)

        '''
        self.state_chains.append(chain)
        buf = [Markov.START_TOKEN] * self.chain_length
        for state in chain:
            v = self.markov.get(tuple(buf), [])
            v.append(state)
            self.markov[tuple(buf)] = v
            buf = buf[1:] + [state]
            self.states.add(state)
        v = self.markov.get(tuple(buf), [])
        v.append(Markov.STOP_TOKEN)
        self.markov[tuple(buf)] = v

    def generate(self, seed=[], max_len=float("inf")):
        '''
        generate a statechain from a (already trained) model
        seed is optional; if provided, will build statechain from seed
        max_len is optional; if provided, will only build statechain up to given len

        note: seed is untested and may very well not work...
        however, seed is core functionality that will help combine all_keys and segmentation (in the future)
        transitions from segments to segments are accomplished by using seed to look for a next valid segment
        that starts with the same seed as the previous segment ends with?
        '''
        buf = self.get_start_buffer(seed)
        state_chain = []
        count = 0

        # we might generate an empty statechain, count will stop us from infinite loop
        while not state_chain and count < 50:
            elem = self.generate_next_state(buf)
            while elem != Markov.STOP_TOKEN and len(state_chain) < max_len:
                state_chain.append(elem)
                buf = self.shift_buffer(buf, elem)
                elem = self.generate_next_state(buf) # generate another
            count += 1
        if not state_chain:
            print("Warning: state_chain empty; seed={}".format(seed))
        return state_chain

    def get_start_buffer(self, seed=[]):
        buf = [Markov.START_TOKEN] * self.chain_length
        if seed and len(seed) > self.chain_length:
            buf = seed[-self.chain_length:]
        elif seed:
            buf[-len(seed):] = seed
        return buf

    def shift_buffer(self, buf, elem):
        return buf[1:] + [elem] # shift buf, add elem to the end

    def generate_next_state(self, buf):
        elem = random.choice(self.markov[tuple(buf)]) # take a random next state using buf
        if elem != Markov.STOP_TOKEN:
            return elem.copy() # prevents change of the underlying states of the markov model
        else:
            return elem

    def copy(self):
        mm = Markov()
        # shallow copies (TODO: deep copy?)
        mm.chain_length = self.chain_length
        mm.markov = {k: v[:] for k, v in self.markov.items()}
        mm.states = self.states.copy()
        mm.state_chains = [ chain[:] for chain in self.state_chains ]
        return mm

    def add_model(self, model):
        '''
        union of the states and state transitions of self and model
        returns a new markov model
        '''
        mm = self.copy()
        for chain in model.state_chains:
            mm.add(chain)
        return mm

def get_key_offset(key1, key2):
    '''
    Returns the relative pitch offset between two keys.
    :param key1: The original key
    :param key2: The new, desired key
    :return: int
    '''
    key1 = str(key1)
    key2 = str(key2)

    # map all minor keys to major keys, and also all redundant major keys to major keys
    key_dict = {'Am': 'C',          # begin minor keys
                'Em': 'G',
                'Bm': 'D',
                'F#/Gbm': 'A',
                'C#/Dbm': 'E',
                'G#/Abm': 'B',
                'D#/Ebm': 'F#',
                'A#/Bbm': 'C#',
                'Dm': 'F',
                'Gm': 'Bb',
                'Cm': 'Eb',
                'Fm': 'Ab',         # end minor keys
                'C#/Db': 'C#',      # begin redundant major keys
                'F#/Gb': 'F#',
                'G#/Ab': 'Ab',
                'D#/Eb': 'Eb',
                'A#/Bb': 'Bb'}      # end redundant major keys

    # keep a dict of the pitch deltas (a delta of 1 corresponds to moving up a half note on the keyboard)
    # transposing from the first key to the second, ie ('C','G') means transposing from C major to G major
    pitch_delta = {('C', 'G'): -5, ('C', 'D'): 2, ('C', 'A'): -3, ('C', 'E'): 4, ('C', 'B'): -1, ('C', 'F#'): 6,
                   ('C', 'C#'): 1, ('C', 'F'): 5, ('C', 'Bb'): -2, ('C', 'Eb'): 3, ('C', 'Ab'): -4,
                   ('G', 'D'): -5, ('G', 'A'): 2, ('G', 'E'): -3, ('G', 'B'): 4, ('G', 'F#'): -1, ('G', 'C#'): 6,
                   ('G', 'F'): -2, ('G', 'Bb'): 3, ('G', 'Eb'): -4, ('G', 'Ab'): 1,
                   ('D', 'A'): -5, ('D', 'E'): 2, ('D', 'B'): -3, ('D', 'F#'): 4, ('D', 'C#'): -1, ('D', 'F'): 3,
                   ('D', 'Bb'): -4, ('D', 'Eb'): 1, ('D', 'Ab'): 6,
                   ('A', 'E'): -5, ('A', 'B'): 2, ('A', 'F#'): -3, ('A', 'C#'): 4, ('A', 'F'): -4, ('A', 'Bb'): 1,
                   ('A', 'Eb'): 6, ('A', 'Ab'): -1,
                   ('E', 'B'): -5, ('E', 'F#'): 2, ('E', 'C#'): -3, ('E', 'F'): 1, ('E', 'Bb'): 6, ('E', 'Eb'): -1,
                   ('E', 'Ab'): 4,
                   ('B', 'F#'): -5, ('B', 'C#'): 2, ('B', 'F'): 6, ('B', 'Bb'): -1, ('B', 'Eb'): 4, ('B', 'Ab'): -3,
                   ('F#', 'C#'): -5, ('F#', 'F'): -1, ('F#', 'Bb'): 4, ('F#', 'Eb'): -3, ('F#', 'Ab'): 2,
                   ('C#', 'F'): 4, ('C#', 'Bb'): -3, ('C#', 'Eb'): 2, ('C#', 'Ab'): -5,
                   ('F', 'Bb'): 5, ('F', 'Eb'): -2, ('F', 'Ab'): 3,
                   ('Bb', 'Eb'): 5, ('Bb', 'Ab'): -2,
                   ('Eb', 'Ab'): 5}

    # remap the keys if needed
    if key1 in key_dict.keys():
        orig_key_major = key_dict[key1]
    else:
        orig_key_major = key1
    if key2 in key_dict.keys():
        new_key_major = key_dict[key2]
    else:
        new_key_major = key2

    if orig_key_major == new_key_major:
        return 0

    if (orig_key_major, new_key_major) in pitch_delta.keys():
        delta = pitch_delta[(orig_key_major, new_key_major)]
    else:   # reverse order is still the same offset, just in the opposite direction
        delta = -1 * pitch_delta[(new_key_major, orig_key_major)]

    return delta

def piece_to_markov_model(musicpiece, classifier=None, segmentation=False, all_keys=False):
    '''
    Train a markov model on a music piece

    Note: (important!)
    If segmentation is True, train a markov model of SegmentStates, each holding a Markov consisting of NoteStates
    Otherwise, the Markov model will consist of NoteStates

    '''

    mm = Markov()
    print("all_keys:" + str(all_keys))
    if not segmentation:
        key_sig, state_chain = NoteState.piece_to_state_chain(musicpiece, True) # always use chords
        offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
        shifted_state_chain = [s.transpose(offset) for s in state_chain]
        mm.add(shifted_state_chain)

        if all_keys: # shift piece up some number of tones, and down some number of tones
            for i in range(1, 6):
                shifted_state_chain = [ s.transpose(i) for s in state_chain ]
                mm.add(shifted_state_chain)
            for i in range(1, 7):
                shifted_state_chain = [ s.transpose(-i) for s in state_chain ]
                mm.add(shifted_state_chain)
    else:
        if classifier == None:
            raise Exception("classifier cannot be None when calling piece_to_markov_model with segmentation=True")
        segmented = experiments.analysis(musicpiece, classifier)
        chosenscore, chosen, labelled_sections = segmented.chosenscore, segmented.chosen, segmented.labelled_sections

        # state_chain implementation #2: more correct than #1, at least
        state_chain = []
        labelled_states = {}
        for ch in chosen:
            i, k = ch[0], ch[1]
            label = labelled_sections[ch]
            ss = labelled_states.get(label, None)
            segment = musicpiece.segment_by_bars(i, i+k)
            if not ss:
                ss = SegmentState(label, piece_to_markov_model(segment, classifier, segmentation=False, all_keys=all_keys))
                labelled_states[label] = ss
            else:
                # ss.mm holds the mm that generates notes
                _state_chain = NoteState.notes_to_state_chain(segment.unified_track.notes, segment.bar)
                ss.mm.add(_state_chain)
            state_chain.append(ss)

        print('Original Sections: ({})'.format(musicpiece.filename))
        print([ g.label for g in state_chain ])
        print(chosenscore)
        mm.add(state_chain)
    return mm

def test_variability(mm, meta, bar):
    '''
    Generate song 10 times from a trained markov model  and print out their lengths
    if they are all the same lengths, chances are the pieces are all the same

    '''
    lens = []
    for i in range(10):
        song, gen, a = generate_song(mm, meta, bar, True)
        lens.append(len(a))
    print(lens)

def generate_song(mm, meta, bar, segmentation=False):

    '''
    Generate music, i.e. a list of MIDI tracks, from the given Markov mm
    you would also need to provide a list of meta events (which you can pull from any MIDI file)

    mm: the Markov model
    meta: list of meta events, is an attribute of class Piece
    bar: ticks per bar
    '''

    song = []
    song.append(meta)

    if not segmentation:
        gen = mm.generate()
        print([g.origin + ('-' if g.chord else '') + g.chord for g in gen])
    else:
        # if segmentation, mm is a markov model of SegmentStates
        # generate SegmentStates from mm and then generate NoteStates from each

        gen_seg = mm.generate()
        print('Rearranged Sections:')
        print([ g.label for g in gen_seg ])
        gen = SegmentState.state_chain_to_note_states(gen_seg)

    a = NoteState.state_chain_to_notes(gen, bar)
    if not a: return generate_song(mm, meta, bar, segmentation)
    song.append([ n.note_event() for n in a ])

    return song, gen, a

def generate_output(args):
    c = patterns.fetch_classifier()
    segmentation = False
    all_keys = False

    if args.mid and args.start is not None and args.end is not None:
        musicpiece = data.piece(args.mid)
        musicpiece = musicpiece.segment_by_bars(args.start, args.end)
        mm = piece_to_markov_model(musicpiece, c, segmentation)
        song, gen, a = generate_song(mm, musicpiece.meta, musicpiece.bar, segmentation)

    else:
        pieces = glob.glob('./mid/Bach/*')
        #pieces = ["mid/Bach/bwv804.mid", "mid/Bach/bwv802.mid"]

        mm = Markov()

        # generate a model _mm for each piece then add them together
        for p in pieces:
            musicpiece = data.piece(p)
            _mm = piece_to_markov_model(musicpiece, c, segmentation, all_keys)
            mm = mm.add_model(_mm)

        song, gen, a = generate_song(mm, musicpiece.meta, musicpiece.bar, segmentation)

    midi.write('output.mid', song)


def generate_score(file):
    '''(str) -> NoneType
    Generate the score (.xml) for the auto-generated midi file.
    Please open using a MusicXMl reader such as Finale NotePad.
    '''
    import music21   # required to display score
    midi_file_output = music21.converter.parse(file)
    midi_file_output.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mid', help='The name of the midi file to use')
    parser.add_argument('-s', '--start', type=int, help='The bar to start from')
    parser.add_argument('-e', '--end', type=int, help='The bar to end at')
    args = parser.parse_args()

    generate_output(args)
    #generate_score()
