import random

from .markov import Markov

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
        self.temporary_klist_counter = 0
        self.temporary_bar_counter = 0
        self.regen_success_counter = 0

    def add(self, chain):
        '''
        Add a statechain to the markov model (i.e. perform training).
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

    def train_obs(self, observations, key_sig):
        '''
        Train the markov-chains of notes corresponding to each chord label.
        '''
        for key in observations:
            bar_markov = {}
            for bar_chain in observations[key]:
                # bar_chains = []
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

                # offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
                # shifted_bar_chain = [s.transpose(offset) for s in state_chain]
                # bar_chains.append(shifted_bar_chain)
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
            self.obs_markov[key] = bar_markov

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
                state_chain.append(elem)
                buf = self.shift_buffer(buf, elem)
                elem = self.generate_next_state(buf)

                # temp_count = 1
                # if elem is HiddenMarkov.STOP_TOKEN and len(state_chain)<min_len:
                #     print('%%%%%%%%%%%%')
                #     print('Start back tracking')
                #     print('%%%%%%%%%%%%')
                #     buf = [state_chain[-temp_count]]
                #     while len(self.markov[tuple(buf)]) < 2:
                #         temp_count += 1
                #         buf = [state_chain[-temp_count]]
                #         print('temp: ', temp_count)
                #     elem = self.generate_next_state(buf)

            count += 1
        if not state_chain:
            print("Warning: state_chain empty; seed={}".format(seed))
        return state_chain

    def generate_bar(self, bar_markov, next_bar_markov, piece_markov, note_chain, bar, next_bar, seed=[], max_len=float("inf")):
        '''
        generate a statechain from a trained model.

        Parameters
        ----------
        bar_markov : dict
        next_bar_markov : dict
        piece_markov : dict
        note_chain : list of NoteState
            The running sequence of states.
        bar : str
            The chord of the current bar.
        next_bar : str or NoneType
            The chord of the next bar. If None, this bar will be treated as the
            end of the piece.
        seed : list of NoteState, optional
            A given seed to build the statechain from.
        max_len : int, optional
            The maximum length of the state chain to build.

        note: seed is untested and may very well not work...
        however, seed is core functionality that will help combine all_keys and segmentation (in the future)
        transitions from segments to segments are accomplished by using seed to look for a next valid segment
        that starts with the same seed as the previous segment ends with?
        '''
        generate_success = False
        forfeit_gen = False
        regen_counter = 0

        while (not generate_success) and (not forfeit_gen):
            klist_success = False
            state_chain = []

            # at the start, note chain is empty
            if not note_chain:
                buf = self.get_start_buffer(seed)
                at_start_of_piece = True
            else: # otherwise let buf be the previous note(s)
                buf = [(note_state, note_state.bar_pos) for note_state in note_chain[-self.chain_length:]]
                at_start_of_piece = False


            # Attempt to generate a statechain, retrying at most 50 times.
            for _ in range(50):
                if state_chain:
                    break
                if at_start_of_piece:
                    # Generate a random next state using buf.
                    elem = self.generate_next_state_bar(buf, bar_markov)
                else:
                    # Otherwise try to connect bar to the previous bar.
                    klist = piece_markov[tuple(buf)]
                    klist = list(set(klist))
                    counter = len(klist) - 1
                    while counter >= 0 and klist:
                        if tuple([klist[counter]]) not in bar_markov:
                            klist.pop(counter)
                        counter -= 1
                    ##elem = mm.generate_next_state(buf)
                    if klist:
                        klist_success = True
                        elem_prime = (random.choice(klist))
                        elem = elem_prime[0]
                    else:
                        buf = self.get_start_buffer(seed)
                        elem = self.generate_next_state_bar(buf,bar_markov)

                while elem != Markov.STOP_TOKEN and len(state_chain) < max_len:
                    state_chain.append(elem)
                    buf = self.shift_bar_buffer(buf, elem)

                    elem = self.generate_next_state_bar(buf,bar_markov)

            if not state_chain:
                print("Warning: state_chain empty; seed={}".format(seed))
                break

            if next_bar != None:
                # check if the end state contain notes allowed in the next bar
                # first by making a list of all possible end states
                last_key_list = []
                last_key_list_final = []
                for key in self.obs_markov[bar]:
                    if HiddenMarkov.STOP_TOKEN in self.obs_markov[bar][key]:
                        last_key_list.append(key)

                for state in last_key_list:
                    if state not in piece_markov:
                        # TODO: Properly figure out how to handle when state is
                        # not in piece_markov, i.e. what it means
                        generate_success = True
                        break
                    slist = piece_markov[state]
                    slist = list(set(slist))
                    counter = len(slist)-1
                    while counter >= 0 and slist:
                        if (slist[counter],) not in next_bar_markov:
                            slist.pop(counter)
                        counter -= 1
                    if len(slist) > 0:
                        last_key_list_final.append(state)

                # print(state_chain)

                if regen_counter >= 50:
                    # impossible to find desired end states, stop trying
                    forfeit_gen = True
                    # TODO: backtracking

                end_states = [(s, s.bar_pos) for s in state_chain[-self.chain_length:]]
                if tuple(end_states) in last_key_list_final:
                    generate_success = True
                    self.regen_success_counter += 1
                    if klist_success == True:
                        self.temporary_klist_counter += 1
                else:
                    regen_counter += 1
            else:
                generate_success = True
                self.regen_success_counter += 1

        return state_chain

    def generate(self, hidden_chain, piece_markov):
        self.temporary_klist_counter = 0
        self.temporary_bar_counter = 0
        self.regen_success_counter = 0

        note_chain = []
        for hs in range(len(hidden_chain)-1):
            bm = self.obs_markov[hidden_chain[hs]]
            nbm = self.obs_markov[hidden_chain[hs + 1]]
            bar = hidden_chain[hs]
            next_bar = hidden_chain[hs+1]
            bar_chain = self.generate_bar(bm, nbm, piece_markov, note_chain, bar, next_bar)
            note_chain = note_chain + bar_chain
            self.temporary_bar_counter += 1

            # print('bar: ', self.temporary_bar_counter)
        bm = self.obs_markov[hidden_chain[-1]]
        nbm = None
        bar = hidden_chain[-1]
        next_bar = None
        bc = self.generate_bar(bm, nbm, piece_markov, note_chain, bar, next_bar)
        note_chain = note_chain + bc
        self.temporary_bar_counter += 1

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
        hmm.markov = {k: v[:] for k, v in self.markov.items()}
        hmm.states = self.states.copy()
        hmm.state_chains = [ chain[:] for chain in self.state_chains ]
        return hmm

    def add_model(self, model):
        '''
        Union of the states and state transitions of self and model.
        Returns a new markov model.
        '''
        hmm = self.copy()
        for chain in model.state_chains:
            hmm.add(chain)
        return hmm
