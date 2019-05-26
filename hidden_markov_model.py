# Markov Model thingy
import argparse
import random, glob
import data, midi, experiments, patterns
from state import NoteState, SegmentState
import pickle
from os.path import basename

temporary_klist_counter = 0
temporary_bar_counter = 0
regen_success_counter = 0

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
        chains =[]
        self.state_chains.append(chain)  # adds a chain to state_chain
        buf = [Markov.START_TOKEN] * self.chain_length  # buf will be ['start_token',...] (repeats self.chain_length times)
        for state in chain:
            v = self.markov.get(tuple(buf), [])
            bp = state.bar_pos
            v.append(tuple([state, bp]))
            #v.append(tuple([state.bar_pos]))
            self.markov[tuple(buf)] = v
            buf = buf[1:] + [tuple([state, bp])] #+ [tuple([state.bar_pos])]
            self.states.add(state)
        v = self.markov.get(tuple(buf), []) #  will either be the value in self.markov with key tuple(buf) (buf should a list of all states now), or []
        v.append(Markov.STOP_TOKEN) # update v by adding STOP_TOKEN to the end
        self.markov[tuple(buf)] = v # update the value that corresponds to key tuple(buf)
        offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
        shifted_chain = [s.transpose(offset) for s in state_chain]
        chains.append(shifted_chain)
        '''
        for i in range(1, 6):
            shifted_state_chain = [ s.transpose(i) for s in state_chain ]
            chains.append(shifted_state_chain)
        for i in range(1, 7):
            shifted_state_chain = [ s.transpose(-i) for s in state_chain ]
            chains.append(shifted_state_chain)
        for chain in chains:
            buf = [HiddenMarkov.START_TOKEN] * self.chain_length  # buf will be ['start_token',...] (repeats self.chain_length times)
            for state in chain:
                v = self.markov.get(tuple(buf), [])
                bp = state.bar_pos
                v.append(tuple([state, bp]))
                #v.append(tuple([state.bar_pos]))
                self.markov[tuple(buf)] = v
                buf = buf[1:] + [tuple([state, bp])] #+ [tuple([state.bar_pos])]
            v = self.markov.get(tuple(buf), [])
            v.append(HiddenMarkov.STOP_TOKEN)
            self.markov[tuple(buf)] = v
        '''

    def generate(self, seed=[], max_len=float("inf"), min_len=float(3000)):
        '''
        generate a statechain from a (already trained) model
        seed is optional; if provided, will build statechain from seed
        len is optional; if provided, will only build statechain up to given len

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
                temp_count = 1
                state_chain.append(elem)
                buf = self.shift_buffer(buf, elem)
                elem = self.generate_next_state(buf)
                print(len(self.markov[tuple(buf)]))
                # back tracks and continue generation if the minimum length is not reached by
                # generation completed
                if elem is Markov.STOP_TOKEN and len(state_chain)<min_len:
                    print('%%%%%%%%%%%%%')
                    print('Start back tracking')
                    print('%%%%%%%%%%%%%')
                    buf = [state_chain[-temp_count]]
                    while len(self.markov[tuple(buf)]) < 2:
                        temp_count += 1
                        buf = [state_chain[-temp_count]]
                        print('temp: ', temp_count)
                    elem = self.generate_next_state(buf)

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
        return buf[1:] + [tuple([elem, elem.bar_pos])]

    def generate_next_state(self, buf):
        elem_prime = random.choice(self.markov[tuple(buf)])
        print("elem_prime: ", elem_prime)
        if elem_prime != HiddenMarkov.STOP_TOKEN:
            elem = elem_prime[0]
            print("elem: ", elem)
            return elem.copy()
        else:
            return elem_prime

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
        key_sig, state_chain, all_bar, observations = NoteState.piece_to_state_chain(musicpiece)

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
    for _ in range(10):
        _, _, a = generate_song(mm, meta, bar, True)
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
    print(a)
    print(len(a))
    song.append([ n.note_event() for n in a ])

    return song, gen, a

def generate_output(args):
    c = patterns.fetch_classifier()
    segmentation = False
    all_keys = False

    if args.mid and args.start is not None and args.end is not None:
        musicpiece = data.Piece(args.mid)
        musicpiece = musicpiece.segment_by_bars(args.start, args.end)
        mm = piece_to_markov_model(musicpiece, c, segmentation)
        song, gen, a = generate_song(mm, musicpiece.meta, musicpiece.bar, segmentation)

    else:
        pieces = glob.glob('./mid/Bach/*')

        mm = Markov()

        # generate a model _mm for each piece then add them together
        p = pieces.pop(0)
        musicpiece = data.Piece(p)
        # check if the markov model has already been built
        noext = basename(musicpiece.filename)
        filename = 'markov_cached/markov-{}.pkl'.format(noext)
        try:
            f = open(filename, 'r')
            _mm = pickle.load(f)
            print('Found markov model')
        except Exception:
            print('Markov model not found. Building it...')
            _mm = piece_to_markov_model(musicpiece, c, segmentation, all_keys) # no transpose
            save = _mm
            f = open(filename, 'w')
            pickle.dump(save, f)


        mm = mm.add_model(_mm)

        for p in pieces:
            musicpiece = data.Piece(p)
            # check if the markov model has already been built
            noext = basename(musicpiece.filename)
            filename = 'markov_cached/markov-{}.pkl'.format(noext)
            try:
                f = open(filename, 'r')
                _mm = pickle.load(f)
                print('Found markov model')
            except Exception:
                print('Markov model not found. Building it...')
                _mm = piece_to_markov_model(musicpiece, c, segmentation, all_keys) # no transpose
                save = _mm
                f = open(filename, 'w')
                pickle.dump(save, f)
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


class HiddenMarkov:
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
        self.obs_markov = {}

    def add(self, chain):
        '''
        add a statechain to the markov model (i.e. perform training)

        '''
        self.state_chains.append(chain)
        buf = [HiddenMarkov.START_TOKEN] * self.chain_length
        for state in chain:
            v = self.markov.get(tuple(buf), [])
            v.append(state)
            self.markov[tuple(buf)] = v
            buf = buf[1:] + [state]
            self.states.add(state)
        v = self.markov.get(tuple(buf), [])
        v.append(HiddenMarkov.STOP_TOKEN)
        self.markov[tuple(buf)] = v

    def train_obs(self, observations, key0_sig):
        ''' train the markov-chains of notes corresponding to each chord label
        '''
        for key in observations:
            bar_markov = {}
            for bar_chain in observations[key]:
                bar_chains = []
                buf = [HiddenMarkov.START_TOKEN] * self.chain_length
                for state in bar_chain:
                    v = bar_markov.get(tuple(buf), [])
                    bp = state.bar_pos
                    v.append(tuple([state, bp]))
                    bar_markov[tuple(buf)] = v
                    buf = buf[1:] + [tuple([state, bp])]
                v = bar_markov.get(tuple(buf), [])
                v.append(HiddenMarkov.STOP_TOKEN)
                bar_markov[tuple(buf)] = v
                '''
                offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
                shifted_bar_chain = [s.transpose(offset) for s in state_chain]
                bar_chains.append(shifted_bar_chain)
                '''
                '''
                for i in range(1, 6):
                    shifted_bar_chain = [ s.transpose(i) for s in state_chain ]
                    bar_chains.append(shifted_bar_chain)
                for i in range(1, 7):
                    shifted_bar_chain = [ s.transpose(-i) for s in state_chain ]
                    bar_chains.append(shifted_bar_chain)
                for chain in bar_chains:
                    buf = [HiddenMarkov.START_TOKEN] * self.chain_length  # buf will be ['start_token',...] (repeats self.chain_length times)
                    for state in chain:
                        v = bar_markov.get(tuple(buf), [])
                        bp = state.bar_pos
                        v.append(tuple([state, bp]))
                        #v.append(tuple([state.bar_pos]))
                        bar_markov[tuple(buf)] = v
                        buf = buf[1:] + [tuple([state, bp])] #+ [tuple([state.bar_pos])]
                    v = bar_markov.get(tuple(buf), [])
                    v.append(HiddenMarkov.STOP_TOKEN)
                    bar_markov[tuple(buf)] = v
                '''
            self.obs_markov[tuple([key])] = bar_markov

    def generate_hidden(self, seed=[], max_len=float("inf"), min_len=float(30)):
        '''
        generate a chain of Hidden States, which are the chord label of bars
        '''
        buf = self.get_start_buffer(seed)
        state_chain = []
        count = 0

        while not state_chain and count < 50:
            elem = self.generate_next_state(buf)
            while elem != HiddenMarkov.STOP_TOKEN and len(state_chain) < max_len:
                temp_count = 1
                state_chain.append(elem)
                buf = self.shift_buffer(buf, elem)
                elem = self.generate_next_state(buf)

                '''
                if elem is HiddenMarkov.STOP_TOKEN and len(state_chain)<min_len:
                    print '%%%%%%%%%%%%%'
                    print 'Start back tracking'
                    print '%%%%%%%%%%%%%'
                    buf = [state_chain[-temp_count]]
                    while len(self.markov[tuple(buf)]) < 2:
                        temp_count += 1
                        buf = [state_chain[-temp_count]]
                        print 'temp: ', temp_count
                    elem = self.generate_next_state(buf)
                '''
            count += 1
        if not state_chain:
            print("Warning: state_chain empty; seed={}".format(seed))
        return state_chain

    def generate_bar(self, bar_markov, next_bar_markov, piece_markov, note_chain, bar, next_bar, seed=[], max_len=float("inf"), min_len=float(2)):
        '''
        generate a statechain from a (already trained) model
        seed is optional; if provided, will build statechain from seed
        len is optional; if provided, will only build statechain up to given len

        note: seed is untested and may very well not work...
        however, seed is core functionality that will help combine all_keys and segmentation (in the future)
        transitions from segments to segments are accomplished by using seed to look for a next valid segment
        that starts with the same seed as the previous segment ends with?
        '''

        global temporary_klist_counter
        global regen_success_counter

        generate_success = False
        forfeit_gen = False
        regen_counter = 0

        while (not generate_success) and (not forfeit_gen):
            klist_success = False
            ##markov = bar_markov
            state_chain = []
            count = 0

            # at the start, note chain is empty
            start_marker = 0
            if not note_chain:
                buf = self.get_start_buffer(seed)
                start_marker = 0
            else: # otherwise let buf be the previous note
                buf = [tuple([note_chain[-1], (note_chain[-1]).bar_pos])]
                start_marker = 1


            # we might generate an empty statechain, count will stop us from infinite loop
            while not state_chain and count < 50:
                # if it is the start of the entire piece
                if start_marker == 0:
                    elem = self.generate_next_state_bar(buf,bar_markov) # generate a random next state using buf
                # otherwise try to connect bar to the previous bar
                else:
                    klist = piece_markov[tuple(buf)]
                    klist = list(set(klist))
                    ##print 'klist len: ', len(klist)
                    counter = len(klist)-1
                    while counter >= 0 and klist:
                        ##print 'counter: ', counter
                        if tuple([klist[counter]]) not in bar_markov:
                            klist.pop(counter)
                            counter = counter - 1
                        else:
                            counter -= 1
                    ##elem = mm.generate_next_state(buf)
                    if klist:

                        klist_success = True
                        elem_prime = (random.choice(klist))
                        elem = elem_prime[0]

                        print(len(klist))
                    else:
                        buf = self.get_start_buffer(seed)
                        elem = self.generate_next_state_bar(buf,bar_markov)

                while elem != Markov.STOP_TOKEN and len(state_chain) < max_len:
                    temp_count = 1
                    state_chain.append(elem)
                    buf = self.shift_bar_buffer(buf, elem)

                    elem = self.generate_next_state_bar(buf,bar_markov)

                count += 1

            if not state_chain:
                print("Warning: state_chain empty; seed={}".format(seed))
                break

            if next_bar != None:
                # check if the end state contain notes allowed in the next bar
                # first by making a list of all possible end states
                last_key_list = []
                last_key_list_final = []
                for key in self.obs_markov[tuple([bar])]:
                    if 'stop_token' in self.obs_markov[tuple([bar])][key]:
                        last_key_list.append(key)

                for state in last_key_list:
                    slist = piece_markov[state]
                    slist = list(set(slist))
                    counter = len(slist)-1
                    while counter >= 0 and slist:
                        if tuple([slist[counter]]) not in next_bar_markov:
                            slist.pop(counter)
                            counter = counter - 1
                        else:
                            counter -= 1
                    if len(slist) > 0:
                        last_key_list_final.append(state)

                print(state_chain)

                if regen_counter >= 50:
                    # impossible to find desired end states, stop trying
                    forfeit_gen = True
                    # TODO: backtracking

                if tuple([tuple([state_chain[-1], (state_chain[-1]).bar_pos])]) in last_key_list_final:
                    generate_success = True
                    regen_success_counter += 1
                    if klist_success == True:
                        temporary_klist_counter += 1
                else:
                    regen_counter += 1
            else:
                generate_success = True
                regen_success_counter += 1

        return state_chain

    def generate(self, hidden_chain, piece_markov):
        global temporary_bar_counter

        note_chain = []
        for hs in range(len(hidden_chain)-1):
            bm = hmm.obs_markov[tuple([hidden_chain[hs]])]
            nbm = hmm.obs_markov[tuple([hidden_chain[hs+1]])]
            bar = hidden_chain[hs]
            next_bar = hidden_chain[hs+1]
            bc = hmm.generate_bar(bm, nbm, piece_markov, note_chain, bar, next_bar)
            note_chain = note_chain + bc
            temporary_bar_counter += 1

            print('bar: ', temporary_bar_counter)
        bm = hmm.obs_markov[tuple([hidden_chain[-1]])]
        nbm = None
        bar = hidden_chain[-1]
        next_bar = None
        bc = hmm.generate_bar(bm, nbm, piece_markov, note_chain, bar, next_bar)
        note_chain = note_chain + bc
        temporary_bar_counter += 1

        return note_chain

    def get_start_buffer(self, seed=[]):
        buf = [HiddenMarkov.START_TOKEN] * self.chain_length
        if seed and len(seed) > self.chain_length:
            buf = seed[-self.chain_length:]
        elif seed:
            buf[-len(seed):] = seed
        return buf

    def shift_buffer(self, buf, elem):
        return buf[1:] + [elem]

    def shift_bar_buffer(self, buf, elem):
        return buf[1:] + [tuple([elem, elem.bar_pos])]

    def generate_next_state(self, buf):
        ''' Generate the next Hidden State '''
        elem = random.choice(self.markov[tuple(buf)])
        if elem != HiddenMarkov.STOP_TOKEN:
            return elem.copy() # prevents change of the underlying states of the markov model
        else:
            return elem

    def generate_next_state_bar(self, buf, markov):
        ''' Generate the note state in a bar '''
        elem_prime = random.choice(markov[tuple(buf)])
        if elem_prime != HiddenMarkov.STOP_TOKEN:
            elem = elem_prime[0]
            return elem.copy() # prevents change of the underlying states of the markov model
        else:
            return elem_prime

    def copy(self):
        hmm = HiddenMarkov()
        # shallow copies (TODO: deep copy?)
        hmm.chain_length = self.chain_length
        hmm.markov = {k: v[:] for k, v in self.markov.iteritems()}
        hmm.states = self.states.copy()
        hmm.state_chains = [ chain[:] for chain in self.state_chains ]
        return hmm

    def add_model(self, model):
        '''
        union of the states and state transitions of self and model
        returns a new markov model
        '''
        hmm = self.copy()
        for chain in model.state_chains:
            hmm.add(chain)
        return hmm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mid', help='The name of the midi file to use')
    parser.add_argument('-s', '--start', type=int, help='The bar to start from')
    parser.add_argument('-e', '--end', type=int, help='The bar to end at')
    parser.add_argument('-o', '--output', help='The name of the generated file')
    args = parser.parse_args()

    c = patterns.fetch_classifier()
    p = [args.mid or "./mid/Bach/bwv802.mid"]
    hmm = HiddenMarkov()
    mm = Markov()

    for piece in p:
        musicpiece = data.Piece(piece)
        # here all_bar is the list of chord labels generated for each bar
        key_sig, state_chain, all_bar, observations = NoteState.piece_to_state_chain(musicpiece)
        hmm.add(all_bar)
        hmm.train_obs(observations, key_sig)
        mm.add(state_chain)

    hidden_chain = hmm.generate_hidden()
    if len(hidden_chain) < 100:
        print('result too short')

    note_chain = hmm.generate(hidden_chain, mm.markov)
    a = NoteState.state_chain_to_notes(note_chain, musicpiece.bar)
    song = []
    song.append(musicpiece.meta)
    song.append([ n.note_event() for n in a ])
    print('bar number: ', temporary_bar_counter)
    print('klist succeeded: {} times'.format(temporary_klist_counter))
    print('regen_success: ', regen_success_counter)
    midi.write(args.output or 'output_hmm.mid', song)
