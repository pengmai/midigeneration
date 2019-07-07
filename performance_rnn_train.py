import torch
from torch import nn
from torch import optim

import numpy as np

from models import PerformanceRNN
from models.performance_rnn import device
from preprocess.create_dataset import run_pipeline

def train():
    loss_function = nn.CrossEntropyLoss()
    model = PerformanceRNN(
        event_dim=388,
        control_dim=1,
        init_dim=32,
        hidden_dim=512)
    optimizer = optim.Adam(model.parameters())
    sequences = run_pipeline()
    events = sequences.reshape(-1, 1)

    def save_model():
        torch.save({'model_state': model.state_dict(),
                    'model_optimizer_state': optimizer.state_dict()},
                    'performance_rnn.sess')

    for i in range(10):
        events = torch.LongTensor(events).to(device)
        init = torch.randn(1, model.init_dim).to(device)
        outputs = model.generate(
            init,
            events.shape[0],
            events=events,
            output_type='logit')

        loss = loss_function(outputs.view(-1, 388), events.view(-1))
        model.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        print(f'iter {i + 1}, loss: {loss.item()}')
        save_model()

if __name__ == '__main__':
    train()