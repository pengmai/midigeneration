"""A simple count-based N-gram model."""

from collections import defaultdict
import numpy as np
from tqdm import tqdm

def nwise(iterable, k=2):
    """Return every consecutive k-group from iterable."""
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    return zip(*[iterable[i:] for i in range(k)])


def sample(dist):
    """Samples from the given 1-D softmax distribution."""
    return np.where(np.cumsum(dist) > np.random.sample())[0][0]


class NGramModel(object):
    """A sequence generation model based off of counting n-grams."""
    def __init__(self, context, vocab_size=388, start_token=355, delta=0):
        self.context = context
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.delta = delta
        self.counts = defaultdict(lambda: np.repeat(0, vocab_size))
        self.normalized = {}

    def train(self, data, verbose=True):
        """Train an n-gram model off of a given dataset."""
        # Append the relevant start tokens to the given dataset
        starts = np.ones((data.shape[0], self.context)).astype(int) * self.start_token
        if data.ndim == 2:
            data = np.hstack((starts, data))
        elif data.ndim == 1:
            data = [np.concatenate((s, d)) for s, d in zip(starts, data)]

        dataset = tqdm(data) if verbose else data
        for piece in dataset:
            for ngram in nwise(piece, k=self.context + 1):
                self.counts[ngram[:self.context]][ngram[-1]] += 1

        self.counts = dict(self.counts)
        for ngram, counts in self.counts.items():
            smoothed = counts + self.delta
            self.normalized[ngram] = smoothed / np.sum(smoothed)

    def generate(self, num_steps, verbose=True):
        """Generate a new sequence by sampling from the training counts."""
        steps = tqdm(range(num_steps)) if verbose else range(num_steps)
        output = [self.start_token] * self.context
        for i in steps:
            key = tuple(output[-self.context:])
            if key in self.normalized:
                next_softmax = self.normalized[key]
                next_token = sample(next_softmax)
                output.append(next_token)
            else:
                print(f'WARNING: Hit zero at step {i}')
                return output
        return output

    def log_likelihood(self, pieces, verbose=True):
        """
        Compute the log likelihood that the given pieces were generated by this model.
        """
        def likelihood(piece):
            likelihood = 0
            for ngram in nwise(piece, k=self.context + 1):
                context, next_token = ngram[:-1], ngram[-1]
                key = tuple(context)
                if key not in self.normalized or self.normalized[key][next_token] == 0:
                    print('WARNING: Encountered likelihood of 0')
                    return -np.inf
                likelihood += np.log(self.normalized[context][next_token])
            return likelihood
        return [likelihood(p) for p in pieces]
