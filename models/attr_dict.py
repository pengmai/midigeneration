"""A convenient data store to access model configuration."""

class AttrDict(dict):
    def __init__(self):
        self.__dict__ = self

_config = {
    'vocab_size': 388,
    'hidden_size': 512,
    'feed_forward_size': 2048,
    'num_layers': 6, # IMPORTANT: Remember to change this back to 6
    'max_relative_position': 100,
    'num_heads': 8,
    'dropout': 0.1,
    'batch_size': 1,
    'num_epochs': 1
}

default_config = AttrDict()
default_config.update(_config)