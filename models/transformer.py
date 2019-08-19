import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from .attention_layers import (
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""

    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(batch_size, -1, -1)  # b x ls x ls
    return subsequent_mask.to(device)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = PositionwiseFeedForward(config)

    def forward(self, inputs, mask=None, cache=None):
        """Forward pass of a single decoder layer."""
        contexts, attention_weights = self.self_attn(
            inputs, inputs, inputs, mask=mask, cache=cache)
        contexts = self.mlp(contexts)
        if cache is not None:
            seq_len = inputs.shape[1]
            cache[:, seq_len - 1] = contexts
            return cache[:, :seq_len], attention_weights
        return contexts, attention_weights


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.feed_forward_size = config.feed_forward_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        assert config.attention_type in ['dot_product', 'dot_product_relative']
        self.use_position_enc = config.attention_type != 'dot_product_relative'
        if self.use_position_enc:
            self.position_enc = PositionalEncoding(self.hidden_size, config.dropout)

        self.num_layers = config.num_layers

        self.layers = nn.ModuleList([
            DecoderLayer(config) for i in range(self.num_layers)])

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, return_attention=True, cache=None):
        """
        Forward pass of the Transformer Decoder.
        Parameters
        ----------
        inputs : (batch_size, seq_len)
            The sequence that has been generated so far.
        return_attention : bool
            If True, return attention weights.
        cache : (num_layers, batch_size, seq_len, hidden_size)
            Cached computations of multi-head attention. Used to accelerate
            generating.
        """
        # Don't bother masking if intermediate computations are cached.
        subsequent_mask = get_subsequent_mask(inputs) if cache is None else None

        # batch_size x seq_len x hidden_size
        embed = self.embedding(inputs)
        if self.use_position_enc:
            embed = self.position_enc(embed)

        self_attention_weights_list = []
        contexts = embed
        for i, layer in enumerate(self.layers):
            contexts, self_attention = layer(
                contexts,
                mask=subsequent_mask,
                cache=None if cache is None else cache[i])
            if return_attention:
                self_attention_weights_list.append(self_attention)

        output = self.out(contexts)

        if return_attention:
            self_attention_weights = torch.stack(self_attention_weights_list)
            return output, self_attention_weights
        return output

    def _initialize_cache(self, batch_size, seq_len):
        """
        Initialize the cache for intermediate computations between layers. Used
        for decoding.
        """
        return torch.zeros(
            self.num_layers,
            batch_size,
            seq_len,
            self.hidden_size)

    def generate(self, primer, steps=500, verbose=True):
        """
        Generates a sequence of steps length from the given primer, taking the
        most likely next word at each step.
        """
        outputs = primer.to(device)
        batch_size, primer_len = primer.shape
        cache = self._initialize_cache(batch_size, primer_len + steps)

        range_iter = range(steps)
        if verbose:
            range_iter = tqdm(range_iter, desc='Greedy decoding')

        for _ in range_iter:
            decoder_outputs = self.forward(
                outputs,
                return_attention=False,
                cache=cache)
            next_word = F.softmax(decoder_outputs[0, -1, :], dim=0).max(0).indices
            next_word = next_word.view(1, 1)
            outputs = torch.cat((outputs, next_word), dim=1)
        return outputs

    def beam_search(self, primer, beam_width, steps=500, verbose=True):
        """Generate a sequence using beam search."""
        assert self.vocab_size >= beam_width

        batch_size, primer_len = primer.shape
        cache = self._initialize_cache(batch_size, primer_len + steps)
        # The saved top beam_width candidates
        beam = torch.zeros(batch_size, beam_width, primer_len + steps).long().to(device)
        beam[:, :, :primer_len] = primer.view(batch_size, 1, primer_len).repeat(1, beam_width, 1)
        # Scores of the top beam_width candidates
        score = torch.zeros(batch_size, beam_width).to(device)

        step_iter = range(primer_len, primer_len + steps)
        if verbose:
            step_iter = tqdm(step_iter)

        for step in step_iter:
            # (batch_size * beam_width, seq_len)
            inputs = beam[:, :, :step].view(-1, step)
            output = self.forward(inputs, return_attention=False, cache=cache)
            # (batch_size * beam_width, vocab_size)
            output = F.log_softmax(output[:, -1, :], dim=1)

            # Consider only the top beam_width candidates for the next token.
            # This is actually a deviation from the full beam search algorithm -
            # traditional beam search would consider vocab_size * beam_width
            # candidates.

            # both top_v and top_i are (batch_size * beam_width, beam_width)
            top_v, top_i = output.topk(beam_width, dim=-1)
            top_v += score.view(batch_size * beam_width, 1)
            # top_v and top_i are (batch_size, beam_width * beam_width)
            top_v = top_v.view(batch_size, -1)
            top_i = top_i.view(batch_size, -1)

            # Find the best beam_width candidates overall
            _, bbi = top_v.topk(beam_width, dim=-1)
            # bbi is (batch_size, beam_width)
            bbi = bbi.view(batch_size, -1)
            bi = torch.arange(batch_size).view(batch_size, 1)
            # Used to choose the original candidate that the new candidates came from
            i = bbi / beam_width

            # Update our running totals.
            score = top_v[bi, bbi]
            beam[:, :, :step] = beam[bi, i, :step]
            event = top_i[bi, bbi]
            beam[bi, torch.arange(beam_width), step] = event

        best = beam[torch.arange(batch_size), score.argmax(-1)]
        return best

    def generate_random(self, primer, steps=500, verbose=True):
        """Like generate, but randomly samples from the softmax distribution."""
        def sample(softmax):
            return torch.distributions.Categorical(softmax).sample()

        range_iter = range(steps)
        if verbose:
            range_iter = tqdm(range_iter, desc='Sampling to decoding')

        outputs = primer.to(device)
        for _ in range_iter:
            decoder_outputs = self.forward(outputs, return_attention=False)
            activations = F.softmax(decoder_outputs[0, -1, :], dim=0)
            next_word = sample(activations).view(1, 1)
            outputs = torch.cat((outputs, next_word), dim=1)
        return outputs

    def log_likelihood(self, pieces):
        """
        Compute the log-likelihood that the given corpus was generated by the model
        over a batch. This tensor can be summed to get the log-likelihood of the
        whole batch/corpus.

        Parameters
        ----------
        pieces : (batch_size, seq_len)
        """
        seq_len = pieces.shape[1] - 1

        out = self.forward(pieces[:, :-1], return_attention=False)
        # normalized: (batch_size, seq_len, vocab_size)
        normalized = F.log_softmax(out, dim=2)

        # Collect the probabilities at the indices of the piece.
        probs = torch.gather(
            normalized,
            dim=2,
            index=pieces[:,1:].view(-1, seq_len, 1)
            ).view(-1, seq_len)
        return torch.sum(probs, dim=1)