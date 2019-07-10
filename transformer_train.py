"""Train a transformer model."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from models.label_smoothing import LabelSmoothing
from models.noam_opt import get_standard_optimizer
from models.nmt_transformer import TransformerDecoder

VOCAB_SIZE = 388
HIDDEN_SIZE = 512

model = TransformerDecoder(VOCAB_SIZE, HIDDEN_SIZE, 8)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothing(size=VOCAB_SIZE, padding_idx=0, smoothing=0.1)
optimizer = get_standard_optimizer(model, HIDDEN_SIZE)

class Batch:
    """Object to hold a batch of data with a label."""
    def __init__(self, sequence):
        """
        Parameters
        ----------
        sequence : (batch_size x seq_length)
        """
        self.sequence = sequence[:, :-1] # Chop off the last token
        self.target = sequence[:, 1:] # Get everything after the first token

def save_model(path):
    torch.save({'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()},
               path)

def load_model(path):
    state = torch.load(path)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['model_optimizer_state'])

def run_epoch(data_iter, model, loss_compute):
    epoch_loss = 0
    total_tokens = 0
    for batch in data_iter:
        ntokens = batch.target.size(-1)
        out, _ = model.forward(batch.sequence)
        loss = loss_compute(out, batch.target, ntokens)
        total_tokens += ntokens
        epoch_loss += loss
    return epoch_loss / total_tokens

# data = np.load('datasets/unravel.npy')
# data = data
# data = torch.from_numpy(data).unsqueeze(-2)
# batch = Batch(data)

# Generate
load_model('nmt-transformer.sess')
print('Generating')
generated = model.generate()[0]
print(generated)
from preprocess.create_dataset import encoded_vals_to_midi_file
encoded_vals_to_midi_file(generated, 'nmt-test.mid')

# Train
# print('Beginning training')
# for epoch in tqdm(range(200), desc='Training'):
#     model.train()
#     out, _ = model(batch.sequence)
#     out_flatten = out.view(-1, out.size(2))
#     targets_flatten = batch.target.view(-1)
#     loss = criterion(out_flatten, targets_flatten)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     tqdm.write(f'Training - Iter: {epoch + 1} loss: {loss}')
# save_model('nmt-transformer.sess')
