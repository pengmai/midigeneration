import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.num_layers = num_layers

        self.self_attentions = nn.ModuleList([CausalScaledDotAttention(
                                    hidden_size=hidden_size,
                                 ) for i in range(self.num_layers)])
        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                 ) for i in range(self.num_layers)])
        self.out = nn.Linear(hidden_size, vocab_size)

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
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        self_attention_weights_list = []
        contexts = embed
        for i in range(self.num_layers):
            new_contexts, self_attention_weights = self.self_attentions[i](contexts, contexts, contexts)
            residual_contexts = contexts + new_contexts
            new_contexts = self.attention_mlps[i](new_contexts)
            contexts = residual_contexts + new_contexts

            if return_attention:
                self_attention_weights_list.append(self_attention_weights)

        output = self.out(contexts)

        if return_attention:
            self_attention_weights = torch.stack(self_attention_weights_list)
            return output, self_attention_weights
        return output

    def generate(self, inputs=torch.LongTensor([[355]]), steps=500):
        """
        Converts the given input sequence to an output sequence.
        """
        outputs = inputs
        for _ in tqdm(range(steps), desc='Greedy decoding'):
            decoder_outputs = self.forward(outputs, return_attention=False)
            generated_words = F.softmax(decoder_outputs, dim=2).max(2).indices
            next_word = generated_words[0][-1].view(1, 1)
            outputs = torch.cat((outputs, next_word), dim=1)
        return outputs.numpy()


class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        # pylint: disable=not-callable
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = torch.tensor(-1e7)

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype=torch.float))

    def forward(self, queries, keys, values):
        """
        The forward pass of the scaled dot attention mechanism.

        Parameters
        ----------
        queries : (batch_size, (k), hidden_size)
            The current decoder hidden state, 2D or 3D tensor.
        keys : (batch_size, seq_len, hidden_size)
            The encoder hidden states for each step of the input sequence.
        values : (batch_size, seq_len, hidden_size)
            The encoder hidden states for each step of the input sequence.

        Returns
        -------
        context : (batch_size, k, hidden_size)
            weighted average of the values
        attention_weights : (batch_size, seq_len, 1)
            Normalized attention weights for each encoder hidden state. 

        The output is a softmax weighting over the seq_len annotations.
        """
        batch_size, seq_len, _ = keys.shape
        if queries.dim() == 2:
            queries = torch.unsqueeze(queries, 1)

        q = self.Q(queries)
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = torch.bmm(k, q.transpose(1, 2)) * self.scaling_factor
        mask = torch.tril(torch.ones_like(unnormalized_attention) * self.neg_inf, diagonal=-1)
        attention_weights = self.softmax(unnormalized_attention + mask)
        context = torch.bmm(attention_weights.transpose(1, 2), v)
        return context, attention_weights