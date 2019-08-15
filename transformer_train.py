"""Train a transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from models.label_smoothing import LabelSmoothing
from models.noam_opt import get_standard_optimizer
from models.transformer import TransformerDecoder, device
from models.attr_dict import music_config


class PerformanceDataset(Dataset):
    def __init__(self, location, max_len=None):
        self.performances = np.load(location, allow_pickle=True)
        if isinstance(self.performances, list) or self.performances.ndim == 1:
            self.performances = np.expand_dims(self.performances, axis=0)
        if max_len:
            self.performances = [p[:max_len] for p in self.performances]

    def __len__(self):
        return len(self.performances)

    def __getitem__(self, i):
        performance = self.performances[i]
        sample = {'sequence': torch.from_numpy(performance[:-1]).to(device),
                  'target': torch.from_numpy(performance[1:]).to(device)}
        return sample


model = TransformerDecoder(music_config)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothing(size=VOCAB_SIZE, padding_idx=0, smoothing=0.1)
# optimizer = get_standard_optimizer(model, music_config['hidden_size'])
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


def save_model(path):
    torch.save({'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()},
               path)


def load_model(path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    # optimizer.load_state_dict(state['model_optimizer_state'])


# dataset = PerformanceDataset('datasets/unravel.npy', max_len=1024)
    #'datasets/maestro2018-truncated-1024.npy', max_len=50)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
dataloader = []


def train(start=0, num_epochs=1, save_every=1):
    print('Beginning training')
    model.train()
    try:
        for epoch in tqdm(range(start + 1, start + num_epochs + 1), desc='Training'):
            losses = []
            with tqdm(dataloader, desc=f'Epoch {epoch}', leave=False) as t:
                for batch in t:
                    out, _ = model(batch['sequence'])
                    out_flatten = out.view(-1, out.size(2))
                    targets_flatten = batch['target'].contiguous().view(-1)
                    loss = criterion(out_flatten, targets_flatten)
                    losses.append(loss.item())
                    t.set_postfix(batch_loss=loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            tqdm.write(f'Training - Epoch: {epoch} loss: {np.mean(losses)}')
            if epoch % save_every == 0:
                save_model(f'transformer-{epoch}.sess')
    except KeyboardInterrupt:
        save_model(f'interrupted-transformer-{epoch}.sess')
        print('Interrupted, saved model at epoch {epoch}')


def generate(checkpoint, steps, filename='nmt-test.mid'):
    # load_model(checkpoint)
    model.eval()
    print('Generating')
    data = np.load('datasets/unravel.npy')[:5]
    data = torch.from_numpy(data).unsqueeze(0)
    generated = model.beam_search(primer=data, beam_width=3, steps=steps)[0]
    return generated
    # from preprocess.create_dataset import encoded_vals_to_midi_file
    # encoded_vals_to_midi_file(generated, filename)

def log_likelihood(pieces):
    """
    Compute the log-likelihood that the given corpus was generated by the model
    over a batch. This tensor can be summed to get the log-likelihood of the
    whole batch/corpus.

    Parameters
    ----------
    pieces : (batch_size, seq_len)
    """
    model.eval()
    seq_len = pieces.shape[1] - 1
    # normalized: (batch_size, seq_len, vocab_size)
    normalized = F.log_softmax(model.forward(pieces[:, :-1])[0], dim=2)

    # Collect the probabilities at the indices of the piece.
    probs = torch.gather(
        normalized,
        dim=2,
        index=pieces[:,1:].view(-1, seq_len, 1)
        ).view(-1, seq_len)
    return torch.sum(probs, dim=1)

if __name__ == '__main__':
    load_model('curated_v4_tiny_200.sess')
    model.eval()
    x = torch.arange(20).unsqueeze(0)
    full = model(x, return_attention=False)
    step = model.forward_step(x)
    print(full[0,-1, 0])
    print(step[0, 0, 0])
    # piece = dataset[0]['sequence'].unsqueeze(0)
    # print(log_likelihood(piece))
    # generate('', 10)
    # x = torch.arange(12).view(3, 4)
    # print(log_likelihood(x))
    # start = 40
    # if start > 0:
    #     load_model(f'transformer-{start}.sess')
    #     print(f'Restored model from epoch {start}')
    # train(start=start, num_epochs=20, save_every=10)
    # generate('relative transformer 10.sess', 100)
