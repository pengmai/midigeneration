"""Define the sublayers in a decoder layer."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_width = self.hidden_size // self.num_heads

        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.hidden_size)

        if config.attention_type == 'dot_product_relative':
            self.attention = RelativeDotProductAttention(config)
        elif config.attention_type == 'dot_product':
            self.attention = ScaledDotProductAttention(config)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of the multi-head attention mechanism.

        Parameters
        ----------
        q : (batch_size, seq_len, hidden_size)
            queries
        k : (batch_size, seq_len, hidden_size)
            keys
        v : (batch_size, seq_len, hidden_size)
            values
        mask : (batch_size, seq_len, seq_len)
        """
        residual = q
        batch_size, seq_len, _ = q.size()

        # Split off the dimensions of q, k, & v into separate attention heads.
        q = self.Q(q).view(batch_size, seq_len,
                           self.num_heads, self.head_width)
        k = self.K(k).view(batch_size, seq_len,
                           self.num_heads, self.head_width)
        v = self.V(v).view(batch_size, seq_len,
                           self.num_heads, self.head_width)

        # (batch_size, num_heads, seq_len, head_width)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Create batch_size x num_heads identical masks.
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        output, attention = self.attention(q, k, v, mask=mask)

        # Concatenate the results of the multiple attention heads together.
        output = output.transpose(2, 1).contiguous().view(
            batch_size, seq_len, -1)

        output = self.dropout(self.mlp(output))
        output = self.layer_norm(output + residual)

        return output, attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=3)
        # pylint: disable=not-callable
        self.scaling_factor = torch.rsqrt(
            torch.tensor(config.hidden_size, dtype=torch.float))

    def forward(self, q, k, v, mask=None):
        attn = torch.einsum('bhqd,bhkd->bhqk', (q, k)) * self.scaling_factor
        if mask is not None:
            attn = attn.masked_fill(mask, -math.inf)
        attn = self.dropout(self.softmax(attn))
        output = torch.einsum('bhqk,bhkd->bhqd', (attn, v))
        return output, attn


def rel_to_abs(x):
    """
    Helper for RelativeDotProductAttention. Implements the "skew" procedure
    outlined in the Music Transformer paper.
    """
    batch, heads, length, _ = x.shape
    pad = torch.zeros(batch, heads, 1).type_as(x)
    x = F.pad(x, (1, 0)).view(batch, heads, length + 1, length)
    return x[:, :, 1:length + 1, :length]

class RelativeDotProductAttention(nn.Module):
    """
    An efficient dot product attention that incorporates relative positional
    encoding.
    """
    def __init__(self, config):
        super(RelativeDotProductAttention, self).__init__()
        self.max_relative_position = config.max_relative_position
        head_width = config.hidden_size // config.num_heads
        self.relations_keys = nn.Parameter(torch.randn(config.num_heads, self.max_relative_position, head_width))
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(config.dropout)

    def _get_rel_embeddings(self, embed, seq_len):
        """
        Get the matrix of relative embeddings for each attention head.
        Returns
        -------
        The embedding matrix : (num_heads, seq_len, head_width)
        """
        pad_len = max(seq_len - self.max_relative_position, 0)
        start_slice = max(self.max_relative_position - seq_len, 0)
        # Pad the rows < seq_len - max_relative_position with zeroes.
        padded = F.pad(embed, (0, 0, pad_len, 0))
        # Only return the bottom seq_len rows.
        return padded[:, start_slice:, :]

    def forward(self, q, k, v, mask=None):
        logits = torch.einsum('bhqd,bhkd->bhqk', (q, k))
        rel_keys = self._get_rel_embeddings(self.relations_keys, q.shape[2])
        rel_logits = torch.einsum('bhqd,hmd->bhqm', (q, rel_keys))
        logits += rel_to_abs(rel_logits)
        if mask is not None:
            logits = logits.masked_fill(mask, -math.inf)

        weights = self.softmax(logits)
        dropped_weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        return output, weights


class PositionwiseFeedForward(nn.Module):
    """A two layer feed-forward network with residual connections."""

    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(config.hidden_size, config.feed_forward_size)
        self.w2 = nn.Linear(config.feed_forward_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        residual = x
        output = self.w2(F.relu(self.w1(x)))
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
