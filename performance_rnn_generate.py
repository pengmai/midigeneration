import argparse
import torch

from models import PerformanceRNN
from models.performance_rnn import device
from preprocess.create_dataset import encoded_vals_to_midi_file
from preprocess.sequence import ControlSeq

def main(args):
    state = torch.load(args.model, map_location='cpu')
    model = PerformanceRNN(
        event_dim=240,
        control_dim=ControlSeq.dim(),
        init_dim=32,
        hidden_dim=512)
    model.load_state_dict(state['model_state'])
    model.eval()

    init = torch.randn(1, model.init_dim).to(device)
    with torch.no_grad():
        outputs = model.generate(init, 5000, verbose=True)
        # outputs = model.beam_search(init, 1000, 300, verbose=True)
    outputs = outputs.cpu().numpy().T

    for i, output in enumerate(outputs):
        name = f'{args.output}-{i:03d}.mid'
        encoded_vals_to_midi_file(output, name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='The location of the PerformanceRNN checkpoint')
    parser.add_argument('-o', '--output', required=True, help='The base name of the midi files to write')
    args = parser.parse_args()

    main(args)
