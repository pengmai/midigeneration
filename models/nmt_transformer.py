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
        self.position_enc = PositionalEncoding(self.hidden_size, config.dropout)

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
        embed = self.position_enc(embed)

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

    def generate(self, inputs=torch.LongTensor([[355]]), steps=500):
        """
        Converts the given input sequence to an output sequence.
        """
        outputs = inputs.to(device)
        for _ in tqdm(range(steps), desc='Greedy decoding'):
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

        self.attention = ScaledDotProductAttention(config)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.xavier_normal_(self.mlp.weight)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        residual = q
        batch_size, seq_len, _ = q.size()

        # Split off the dimensions of q, k, & v into separate attention heads.
        q = self.Q(q).view(batch_size, seq_len, self.num_heads, self.head_width)
        k = self.K(k).view(batch_size, seq_len, self.num_heads, self.head_width)
        v = self.V(v).view(batch_size, seq_len, self.num_heads, self.head_width)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_width)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        output, attention = self.attention(q, k, v, mask=mask)

        # Concatenate the results of the multiple attention heads together.
        output = output.view(self.num_heads, batch_size, seq_len, self.head_width)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, -1)

        output = self.dropout(self.mlp(output))
        output = self.layer_norm(output + residual)

        return output, attention

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=2)
        # pylint: disable=not-callable
        self.scaling_factor = torch.rsqrt(torch.tensor(config.hidden_size, dtype=torch.float))

    def forward(self, q, k, v, mask=None):
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scaling_factor
        if mask is not None:
            attention = attention.masked_fill(mask, -math.inf)

        attention = self.dropout(self.softmax(attention))
        output = torch.bmm(attention, v)
        return output, attention

class PositionalEncoding(nn.Module):
    """Absolute sinusoidal positional encoding."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)