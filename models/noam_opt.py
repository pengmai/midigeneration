"""The optimizer needed to train the transformer."""

import torch

class NoamOpt:
    """Optimizer wrapper that implements learning rate decay."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        """Clears the gradients of the underlying optimizer."""
        self.optimizer.zero_grad()

    def step(self):
        """Update parameters and learning rate."""
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement learning rate decay."""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        """Serialize the optimizer state to a dictionary."""
        return {'optimizer_state': self.optimizer.state_dict(),
                '_step': self._step,
                'warmup': self.warmup,
                'factor': self.factor,
                'model_size': self.model_size,
                '_rate': self._rate}

    def load_state_dict(self, state_dict):
        """Restore the model state from a state_dict."""
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self._step = state_dict['_step']
        self.warmup = state_dict['warmup']
        self.factor = state_dict['factor']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['_rate']

def get_standard_optimizer(model, d_model):
    """Returns an Adam optimizer with lr decay and standard hyperparameters."""
    return NoamOpt(d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
