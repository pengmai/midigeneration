"""A convenient data store to access model configuration."""

class AttrDict(dict):
    def __init__(self):
        self.__dict__ = self

_config = {
    'vocab_size': 388,
    'hidden_size': 512,
    'feed_forward_size': 2048,
    'num_layers': 8, # In the original paper, this is 6.
    'num_heads': 8,
    'dropout': 0.1
}

default_config = AttrDict()
default_config.update(_config)