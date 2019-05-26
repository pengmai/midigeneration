import chords
from decimal import ROUND_HALF_DOWN, Decimal as fixed

class State(object):

    '''
    Basic interface of a state to be used in a Markov model
    Please override state_data() and copy()

    '''

    def state_data(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def __hash__(self):
        tup = self.state_data()
        return hash(tup)

    def __eq__(self, other):
        return isinstance(other, State) and self.state_data() == other.state_data()

    def __repr__(self):
        tup = self.state_data()
        return str(tup)

    def copy(self):
        raise NotImplementedError("Subclass must implement abstract method")


class SegmentState(State):
    '''
    SegmentState: a Markov state representing a segment of music (from segmentation)

    Instance attributes:
    - string label: name of the SegmentState, possibly arbitrary, for bookkeeping
    - Markov mm: a Markov model consisting of NoteStates. This will be used for generating the NoteStates
        within the segment

    '''

    def __init__(self, label, mm):
        self.label = label
        self.mm = mm

    def state_data(self):
        relevant = [self.label]
        return tuple(relevant)

    def copy(self):
        s = SegmentState(self.label, self.mm)
        return s

    @staticmethod
    def state_chain_to_note_states(state_chain):
        note_states = []
        for s in state_chain:
            gen = s.mm.generate()
            note_states.extend(gen)
        return note_states


class NoteState(State):
    '''
    NoteState: a Markov state representing a group of notes starting from the same position

    Instance attributes:
    - notes: a list of Notes, all with the same position, sorted by duration then pitch
    - bar: number of ticks in a bar (this is used for converting positions to fixed/decimal values
    - bar_pos: a fixed/decimal value denoting the position of these notes relative to a bar

    (docs: todo)
    - state_position:
    - state_duration:
    - chord: the chord that these notes most likely belong to
    - origin: midi filename

    '''

    def __init__(self, notes, bar, chord='', origin='', bar_number=0):
        # State holds multiple notes, all with the same pos
        self.notes = [ n.copy() for n in sorted(notes, key=lambda x: (x.dur, x.pitch)) ]
        self.bar = bar
        self.bar_pos = fixed(self.notes[0].pos % bar) / bar
        self.state_position = fixed(self.notes[0].pos) / bar
        self.state_duration = 0 # set later
        self.chord = chord
        self.origin = origin
        self.bar_number = bar_number

        for n in self.notes:
            n.dur = fixed(n.dur) / bar

    def state_data(self):
        ''' make hashable version of state information intended to be hashed '''
        notes_info = [n.pitch for n in self.notes]
        quantized_pos = self.bar_pos.quantize(fixed('0.01'), rounding=ROUND_HALF_DOWN)
        quantized_dur = self.state_duration.quantize(fixed('0.0001'), rounding=ROUND_HALF_DOWN)
        relevant = [quantized_pos, quantized_dur, self.chord, frozenset(notes_info)]
        return tuple(relevant)

    def copy(self):
        s = NoteState(self.notes, 1, self.chord, self.origin)
        s.bar = self.bar
        s.bar_pos = self.bar_pos
        s.state_position = self.state_position
        s.state_duration = self.state_duration
        return s

    def transpose(self, offset):
        s = self.copy()
        ctemp = self.chord.split('m')[0]
        s.chord = chords.translate(chords.untranslate(ctemp)+offset) + ('m' if 'm' in self.chord else '')
        s.origin = 'T({})'.format(offset) + s.origin
        for n in s.notes:
            n.pitch += offset
        return s

    def to_notes(self, bar, last_pos):
        '''
        Convert this NoteState to a list of notes,
        pos of each note will be assigned to last_pos
        bar is the number of ticks to form a bar
        returns the list of notes and the position of the next state (which can be used for the next call)
        '''

        notes = []
        for n in self.notes:
            nc = n.copy()
            nc.pos = last_pos
            nc.dur = int(n.dur * bar)
            notes.append(nc)
        last_pos += int(self.state_duration * bar)
        return notes, last_pos

    @staticmethod
    def state_chain_to_notes(state_chain, bar):
        '''
        Convert a state chain (a list of NoteStates) to notes
        arg bar: number of ticks to define a bar for midi files

        '''

        last_pos = 0
        notes = []
        for s in state_chain: # update note positions for each s in state_chain
            for n in s.notes:
                nc = n.copy()
                nc.pos = int(last_pos * bar)
                nc.dur = int(n.dur * bar)
                notes.append(nc)
            last_pos += s.state_duration
        return notes

    @staticmethod
    def notes_to_state_chain(notes, bar):
        '''
        Convert a list of Notes to a state chain (list of NoteStates)
        arg bar: number of ticks to define a bar for midi files

        '''

        # group notes into bins by their starting positions
        bin_by_pos = {}
        for n in notes:
            v = bin_by_pos.get(n.pos, [])
            v.append(n)
            bin_by_pos[n.pos] = v

        positions = sorted(bin_by_pos.keys())

        # produce a state_chain by converting the notes at every position x into a NoteState
        state_chain = list(map(lambda x: NoteState(bin_by_pos[x], bar), positions))

        if not len(state_chain):
            return state_chain

        # calculate state_duration for each state
        for i in range(len(state_chain) - 1):
            state_chain[i].state_duration = state_chain[i+1].state_position - state_chain[i].state_position
        state_chain[-1].state_duration = max(n.dur for n in state_chain[-1].notes) # the last state needs special care

        return state_chain

    @staticmethod
    def piece_to_state_chain(piece, use_chords=True):
        '''
        Convert a data.piece into a state chain (list of NoteStates)
        arg use_chords: if True, NoteState holds chord label as state information

        '''
        # TODO: shouldn't have to hardcode this
        use_chords = True
        key_sig = ""
        observations = {}

        # group notes into bins by their starting positions
        bin_by_pos = {}
        for n in piece.unified_track.notes:
            v = bin_by_pos.get(n.pos, [])
            v.append(n)
            bin_by_pos[n.pos] = v

        positions = sorted(bin_by_pos.keys())
        if use_chords:
            cc = chords.fetch_classifier()
            key_sig, allbars = cc.predict(piece) # assign chord label to each bar
            state_chain = list(map(lambda x: NoteState(bin_by_pos[x], piece.bar, chord=allbars[x//piece.bar], origin=piece.filename, bar_number=x//piece.bar), positions))
            for counter in range(len(allbars)):
                bc = observations.get(allbars[counter], [])
                bn = []
                for s in state_chain:
                    if s.bar_number == counter:
                        bn.append(s)
                bc.append(bn)
                observations[allbars[counter]] = bc

        else:
            state_chain = list(map(lambda x: NoteState(bin_by_pos[x], piece.bar, chord='', origin=piece.filename), positions))

        if not len(state_chain):
            return state_chain

        # calculate state_duration for each state
        for i in range(len(state_chain) - 1):
            state_chain[i].state_duration = state_chain[i+1].state_position - state_chain[i].state_position
        state_chain[-1].state_duration = max(n.dur for n in state_chain[-1].notes) # the last state needs special care

        return key_sig, state_chain, allbars, observations

    def __repr__(self):
        tup = self.state_data()
        return str(tup) + ' ' + str(self.notes)
