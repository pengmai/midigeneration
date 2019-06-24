import random

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
        '''Add a statechain to the markov model (i.e. perform training).'''
        # chains = []
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
        # offset = get_key_offset(key_sig[0], 'C')   # transpose everything to C major
        # shifted_chain = [s.transpose(offset) for s in state_chain]
        # chains.append(shifted_chain)
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
        Generate a statechain from a (already trained) model.
        seed is optional; if provided, will build statechain from seed
        len is optional; if provided, will only build statechain up to given len

        note: seed is untested and may very well not work...
        however, seed is core functionality that will help combine all_keys and
        segmentation (in the future). Transitions from segments to segments are
        accomplished by using seed to look for a next valid segment that starts
        with the same seed as the previous segment ends with?
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
                # print(len(self.markov[tuple(buf)]))
                # back tracks and continue generation if the minimum length is not reached by
                # generation completed
                if elem is Markov.STOP_TOKEN and len(state_chain) < min_len:
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
        if elem_prime != Markov.STOP_TOKEN:
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
