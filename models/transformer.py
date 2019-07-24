import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)  # b x ls x ls

    return subsequent_mask

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.feed_forward_size),
            nn.ReLU(),
            nn.Linear(config.feed_forward_size, config.hidden_size))

    def forward(self, inputs, mask=None):
        """Forward pass of a single decoder layer."""
        residual = inputs
        contexts, attention_weights = self.self_attn(inputs, inputs, inputs, mask=mask)
        contexts += residual
        residual = contexts
        contexts = self.mlp(contexts)
        contexts += residual

        return contexts, attention_weights

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.feed_forward_size = config.feed_forward_size

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        self.num_layers = config.num_layers

        self.layers = nn.ModuleList([
            DecoderLayer(config) for i in range(self.num_layers)])

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
        subsequent_mask = get_subsequent_mask(inputs)
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        self_attention_weights_list = []
        contexts = embed
        for layer in self.layers:
            contexts, self_attention = layer(contexts, mask=subsequent_mask)

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
        mean, std = 0, math.sqrt(2.0 / (self.hidden_size + self.head_width))
        nn.init.normal_(self.Q.weight, mean=mean, std=std)
        nn.init.normal_(self.K.weight, mean=mean, std=std)
        nn.init.normal_(self.V.weight, mean=mean, std=std)

        self.attention = RelativeDotProductAttention(config)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.mlp.weight)

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
        q = self.Q(q).view(batch_size, seq_len, self.num_heads, self.head_width)
        k = self.K(k).view(batch_size, seq_len, self.num_heads, self.head_width)
        v = self.V(v).view(batch_size, seq_len, self.num_heads, self.head_width)

        # (batch_size x num_heads, seq_len, head_width)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)
        # v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)

        # Create batch_size x num_heads identical masks.
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        output, attention = self.attention(q, k, v, mask=mask)

        # Concatenate the results of the multiple attention heads together.
        output = output.transpose(2, 1).contiguous().view(batch_size, seq_len, -1)
        # output = output.view(self.num_heads, batch_size, seq_len, self.head_width)
        # output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, -1)

        output = self.dropout(self.mlp(output))
        output = self.layer_norm(output + residual)

        return output, attention

class RelativeDotProductAttention(nn.Module):
    def __init__(self, config):
        super(RelativeDotProductAttention, self).__init__()

        self.max_relative_position = config.max_relative_position
        vocab_size = self.max_relative_position * 2 + 1
        head_width = config.hidden_size // config.num_heads
        self.relations_keys = nn.Embedding(vocab_size, head_width)
        self.relations_vals = nn.Embedding(vocab_size, head_width)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        """Forward pass of the scaled dot product attention with relative
        encoding.

        Parameters:
        -----------
        q, k, v : (batch_size, num_heads, seq_len, head_width)
            The queries, keys, and values to use.
        """
        _, _, seq_len, d_k = q.shape
        rel_positions = self.generate_relative_positions_matrix(seq_len)
        rel_keys = self.relations_keys(rel_positions)
        rel_vals = self.relations_vals(rel_positions)

        logits = self.relative_attention(q, k, rel_keys, transpose=True)
        if mask is not None:
            logits = logits.masked_fill(mask, -math.inf)

        weights = self.softmax(logits)
        dropped_weights = self.dropout(weights)
        output = self.relative_attention(dropped_weights, v, rel_vals, transpose=False)
        return output, weights

    def generate_relative_positions_matrix(self, length):
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat_clipped = torch.clamp(distance_mat,
                                           -self.max_relative_position,
                                           self.max_relative_position)
        return distance_mat_clipped + self.max_relative_position

    def relative_attention(self, x, y, z, transpose=False):
        batch_size, num_heads, seq_len, _ = x.shape
        xy_matmul = torch.matmul(x, y.transpose(-2, -1) if transpose else y)

        x_v = x.permute(2, 0, 1, 3).contiguous().view(seq_len, num_heads * batch_size, -1)
        x_tz = torch.matmul(x_v, z.transpose(-2, -1) if transpose else z) \
            .view(seq_len, batch_size, num_heads, -1) \
            .permute(1, 2, 0, 3)
        return xy_matmul + x_tz
