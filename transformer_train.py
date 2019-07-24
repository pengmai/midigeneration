"""Train a transformer model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from models.label_smoothing import LabelSmoothing
from models.noam_opt import get_standard_optimizer
from models.transformer import TransformerDecoder, device
from models.attr_dict import default_config

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

model = TransformerDecoder(default_config)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothing(size=VOCAB_SIZE, padding_idx=0, smoothing=0.1)
optimizer = get_standard_optimizer(model, default_config['hidden_size'])

def save_model(path):
    torch.save({'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()},
               path)

def load_model(path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['model_optimizer_state'])

dataset = PerformanceDataset('datasets/maestro2018-truncated-1024.npy', max_len=50)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

def train(start=0, num_epochs=1, save_every=1):
    print('Beginning training')
    model.train()
    try:
        for epoch in tqdm(range(start + 1, start + num_epochs + 1), desc='Training'):
            losses = []
            for batch in tqdm(dataloader, desc=f'epoch {epoch}', leave=False):
                out, _ = model(batch['sequence'])
                out_flatten = out.view(-1, out.size(2))
                targets_flatten = batch['target'].contiguous().view(-1)
                loss = criterion(out_flatten, targets_flatten)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tqdm.write(f'Training - Iter: {epoch} loss: {np.mean(losses)}')
            if epoch % save_every == 0:
                save_model(f'transformer-{epoch}.sess')
    except Exception:
        save_model(f'interrupted-transformer-{epoch}.sess')

def generate(checkpoint, steps, filename='nmt-test.mid'):
    load_model(checkpoint)
    model.eval()
    print('Generating')
    data = np.load('datasets/unravel.npy')[:50]
    data = torch.from_numpy(data).unsqueeze(0)
    generated = model.generate(primer=data, steps=steps)[0]
    from preprocess.create_dataset import encoded_vals_to_midi_file
    encoded_vals_to_midi_file(generated, filename)

if __name__ == '__main__':
    # train(num_epochs=1, save_every=1)
    generate('relative transformer 10.sess', 100)
