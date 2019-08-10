import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
from .attention_layers import (
    MultiHeadAttention,
    RelativeMultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    generate_relative_positions_matrix
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(batch_size, -1, -1)  # b x ls x ls

    return subsequent_mask


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = PositionwiseFeedForward(config)

    def forward(self, inputs, mask=None):
        """Forward pass of a single decoder layer."""
        contexts, attention_weights = self.self_attn(
            inputs, inputs, inputs, mask=mask)
        contexts = self.mlp(contexts)

        return contexts, attention_weights


class RelativeDecoderLayer(nn.Module):
    def __init__(self, config):
        super(RelativeDecoderLayer, self).__init__()
        self.self_attn = RelativeMultiHeadAttention(config)
        self.mlp = PositionwiseFeedForward(config)

    def forward(self, inputs, rel_positions, mask=None):
        """Forward pass of a single decoder layer."""
        contexts, attention_weights = self.self_attn(
            inputs, inputs, inputs, rel_positions, mask=mask)
        contexts = self.mlp(contexts)

        return contexts, attention_weights


class AbsoluteTransformerDecoder(nn.Module):
    def __init__(self, config):
        super(AbsoluteTransformerDecoder, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.feed_forward_size = config.feed_forward_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_enc = PositionalEncoding(self.hidden_size, config.dropout)

        self.num_layers = config.num_layers

        self.layers = nn.ModuleList([
            DecoderLayer(config) for i in range(self.num_layers)])

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs):
        subsequent_mask = get_subsequent_mask(inputs)
        batch_size, seq_len = inputs.size()

        # batch_size x seq_len x hidden_size
        embed = self.embedding(inputs)
        embed = self.position_enc(embed)

        self_attention_weights_list = []
        contexts = embed
        for layer in self.layers:
            contexts, self_attention = layer(contexts, mask=subsequent_mask)
            self_attention_weights_list.append(self_attention)

        output = self.out(contexts)

        self_attention_weights = torch.stack(self_attention_weights_list)
        return output, self_attention_weights

    def generate(self, primer=torch.LongTensor([[355]]), steps=500, verbose=True):
        """
        Generates a sequence of steps length from the given primer.
        """
        outputs = primer.to(device)

        range_iter = range(steps)
        if verbose:
            range_iter = tqdm(range_iter, desc='Greedy decoding')

        for _ in range_iter:
            decoder_outputs, _ = self.forward(outputs)
            generated_words = F.softmax(decoder_outputs, dim=2).max(2).indices
            next_word = generated_words[0][-1].view(1, 1)
            outputs = torch.cat((outputs, next_word), dim=1)
        return outputs.cpu().numpy()[0]

    def generate_random(self, primer=torch.LongTensor([[355]]), steps=500):
      """Like generate, but randomly samples from the softmax distribution."""
      def sample(softmax):
          return torch.nonzero(softmax.cumsum(dim=0) > np.random.sample())[0][0]

      outputs = primer.to(device)
      for _ in tqdm(range(steps), desc='Sampling to decode'):
        decoder_outputs, _ = self.forward(outputs)
        activations = F.softmax(decoder_outputs[0, -1, :], dim=0)
        next_word = sample(activations).view(1, 1)
        outputs = torch.cat((outputs, next_word), dim=1)
      return outputs.cpu().numpy()[0]


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.feed_forward_size = config.feed_forward_size
        self.max_relative_position = config.max_relative_position

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.num_layers = config.num_layers

        self.layers = nn.ModuleList([
            RelativeDecoderLayer(config) for i in range(self.num_layers)])

        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inputs, return_attention=True):
        """
        Forward pass of the Transformer decoder.

        Parameters
        ----------
        inputs : (batch_size, decoder_seq_len)
            Input token indexes across a batch for all the time step.

        Returns
        -------
        output: (batch_size, decoder_seq_len, vocab_size)
            Un-normalized scores for each token in the vocabulary,
            across a batch for all the decoding time steps.
        attentions: (batch_size, encoder_seq_len, decoder_seq_len)
            The stacked attention weights applied to the encoder annotations
        """
        batch_size, seq_len = inputs.size()
        subsequent_mask = get_subsequent_mask(inputs)
        rel_positions = generate_relative_positions_matrix(seq_len, self.max_relative_position)
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        self_attention_weights_list = []
        contexts = embed
        for layer in self.layers:
            contexts, self_attention = layer(contexts, rel_positions, mask=subsequent_mask)

            if return_attention:
                self_attention_weights_list.append(self_attention)

        output = self.out(contexts)

        if return_attention:
            self_attention_weights = torch.stack(self_attention_weights_list)
            return output, self_attention_weights
        return output

    def generate(self, primer=torch.LongTensor([[355]]), steps=500, verbose=True):
        """
        Generates a sequence of steps length from the given primer.
        """
        outputs = primer.to(device)

        range_iter = range(steps)
        if verbose:
            range_iter = tqdm(range_iter, desc='Greedy decoding')

        for _ in range_iter:
            decoder_outputs = self.forward(outputs, return_attention=False)
            generated_words = F.softmax(decoder_outputs, dim=2).max(2).indices
            next_word = generated_words[0][-1].view(1, 1)
            outputs = torch.cat((outputs, next_word), dim=1)
        return outputs.cpu().numpy()
