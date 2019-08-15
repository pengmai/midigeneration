"""A convenient data store to access model configuration."""

class AttrDict(dict):
    def __init__(self):
        self.__dict__ = self

_config = {
    'vocab_size': 388,
    'hidden_size': 512,
    'feed_forward_size': 2048,
    'num_layers': 6,
    'max_relative_position': 100, # TODO: Experiment with this
    'num_heads': 8,
    'dropout': 0.1,
    'batch_size': 1,
    'num_epochs': 1
}

_music_config = {
    'vocab_size': 240,
    'hidden_size': 64, # Originally 256
    'feed_forward_size': 256, # Originally 1024
    'max_relative_position': 512
}

default_config = AttrDict()
default_config.update(_config)

music_config = AttrDict()
music_config.update(_config)
music_config.update(_music_config)