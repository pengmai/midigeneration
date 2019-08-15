"""Define the sublayers in a decoder layer."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(RelativeMultiHeadAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_width = self.hidden_size // self.num_heads

        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention = RelativeDotProductAttention(config)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, rel_positions, mask=None):
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

        # (batch_size x num_heads, seq_len, head_width)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Create batch_size x num_heads identical masks.
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        output, attention = self.attention(q, k, v, rel_positions, mask=mask)

        # Concatenate the results of the multiple attention heads together.
        output = output.transpose(2, 1).contiguous().view(
            batch_size, seq_len, -1)

        output = self.dropout(self.mlp(output))
        output = self.layer_norm(output + residual)

        return output, attention


def generate_relative_positions_matrix(length, max_relative_position):
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length).view(length, length)
    distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       -max_relative_position,
                                       max_relative_position)
    return distance_mat_clipped + max_relative_position

def rel_to_abs(x):
    """
    Helper for RelativeDotProductAttentionV2. Implements the "skew" procedure
    outlined in the Music Transformer paper.
    """
    batch, heads, length, _ = x.shape
    pad = torch.zeros(batch, heads, 1).type_as(x)
    x = F.pad(x, (1, 0)).view(batch, heads, length + 1, length)
    return x[:, :, 1:length + 1, :length]


class RelativeDotProductAttentionV2(nn.Module):
    def __init__(self, config):
        super(RelativeDotProductAttentionV2, self).__init__()
        self.max_relative_position = config.max_relative_position
        head_width = config.hidden_size // config.num_heads
        self.relations_keys = nn.Embedding(self.max_relative_position, head_width)
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(config.dropout)

    def _get_rel_embeddings(self, embed, seq_len):
        pad_len = max(seq_len - self.max_relative_position, 0)
        start_slice = max(self.max_relative_position - seq_len, 0)
        padded = F.pad(embed, (0, 0, pad_len, 0))
        return padded[start_slice:, :]

    def forward(self, q, k, v, mask=None):
        logits = torch.einsum('bhqd,bhkd->bhqk', (q, k))
        rel_keys = self._get_rel_embeddings(self.relations_keys.weight, q.shape[2])
        rel_logits = torch.einsum('bhqd,md->bhqm', (q, rel_keys))
        logits += rel_to_abs(rel_logits)
        if mask is not None:
            logits = logits.masked_fill(mask, -math.inf)

        weights = self.softmax(logits)
        dropped_weights = self.dropout(weights)
        output = torch.matmul(weights, v)
        return output, weights


class RelativeDotProductAttention(nn.Module):
    def __init__(self, config):
        super(RelativeDotProductAttention, self).__init__()

        self.max_relative_position = config.max_relative_position
        vocab_size = self.max_relative_position * 2 + 1
        head_width = config.hidden_size // config.num_heads
        self.relations_keys = nn.Embedding(vocab_size, head_width)
        self.relations_vals = nn.Embedding(vocab_size, head_width)
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, rel_positions, mask=None):
        """Forward pass of the scaled dot product attention with relative
        encoding.

        Parameters:
        -----------
        q, k, v : (batch_size, num_heads, seq_len, head_width)
            The queries, keys, and values to use.
        rel_positions : (seq_len, seq_len)
            The relative position matrix to use for the corresponding sequence.
        """
        _, _, seq_len, d_k = q.shape

        # (seq_len, seq_len, head_width)
        rel_keys = self.relations_keys(rel_positions)
        rel_vals = self.relations_vals(rel_positions)

        # (batch_size, num_heads, seq_len, seq_len)
        logits = self.relative_attention(q, k, rel_keys, transpose=True)
        if mask is not None:
            logits = logits.masked_fill(mask, -math.inf)

        weights = self.softmax(logits)
        dropped_weights = self.dropout(weights)
        output = self.relative_attention(
            dropped_weights, v, rel_vals, transpose=False)
        return output, weights

    def relative_attention(self, x, y, z, transpose=False):
        batch_size, num_heads, seq_len, _ = x.shape
        xy_matmul = torch.matmul(x, y.transpose(-2, -1) if transpose else y)

        x_v = x.permute(2, 0, 1, 3).contiguous().view(
            seq_len, num_heads * batch_size, -1)
        x_tz = torch.matmul(x_v, z.transpose(-2, -1) if transpose else z) \
            .view(seq_len, batch_size, num_heads, -1) \
            .permute(1, 2, 0, 3)
        return xy_matmul + x_tz


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

        self.attention = ScaledDotProductAttention(config)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None, only_last=False):
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
        if only_last:
            q = self.Q(q[:, -1, :]).view(batch_size,
                                         self.num_heads, self.head_width)
        else:
            q = self.Q(q).view(batch_size, seq_len,
                               self.num_heads, self.head_width)
        k = self.K(k).view(batch_size, seq_len,
                           self.num_heads, self.head_width)
        v = self.V(v).view(batch_size, seq_len,
                           self.num_heads, self.head_width)

        # (batch_size x num_heads, seq_len, head_width)
        if only_last:
            q = q.view(-1, self.head_width)
        else:
            q = q.permute(2, 0, 1, 3).contiguous(
            ).view(-1, seq_len, self.head_width)
        k = k.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_len, self.head_width)
        v = v.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_len, self.head_width)

        if only_last:
            output = self.attention.forward_step(q, k, v)
            output = self.mlp(output.view(batch_size, -1))
            output = self.layer_norm(output + residual[:, -1, :])
            return output

        # Create batch_size x num_heads identical masks.
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        output, attention = self.attention(q, k, v, mask=mask)

        # Concatenate the results of the multiple attention heads together.
        output = output.view(self.num_heads, batch_size,
                             seq_len, self.head_width)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_len, -1)

        output = self.dropout(self.mlp(output))
        output = self.layer_norm(output + residual)

        return output, attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=2)
        # pylint: disable=not-callable
        self.scaling_factor = torch.rsqrt(
            torch.tensor(config.hidden_size, dtype=torch.float))

    def forward(self, q, k, v, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scaling_factor
        if mask is not None:
            attention = attention.masked_fill(mask, -math.inf)

        attention = self.dropout(self.softmax(attention))
        output = torch.bmm(attention, v)
        return output, attention

    def forward_step(self, q, k, v):
        """
        Compute attention for the last token without computing other attentions.
        Used for beam search.

        Parameters
        ----------
        q : (batch_size, head_width)
            Query for the last token
        k : (batch_size, seq_len, head_width)
            All keys
        v : (batch_size, seq_len, head_width)
            All values
        """
        # Only multiply the last query
        attention = torch.einsum('bd,bld->bl', (q, k)) * self.scaling_factor
        attention = F.softmax(attention, dim=1)
        output = torch.einsum('bl,bld->bd', (attention, v))
        return output


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
    """Implement the PE function."""

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
