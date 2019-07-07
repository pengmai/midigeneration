import torch

from models import PerformanceRNN
from models.performance_rnn import device
from preprocess.create_dataset import encoded_vals_to_midi_file

state = torch.load('undertale_checkpoints/performance_rnn_80.sess', map_location='cpu')
model = PerformanceRNN(
    event_dim=388,
    control_dim=1,
    init_dim=32,
    hidden_dim=512)
model.load_state_dict(state['model_state'])
model.eval()

init = torch.randn(4, model.init_dim).to(device)

with torch.no_grad():
    outputs = model.generate(init, 1000, verbose=True)
    # outputs = model.beam_search(init, 1000, 3, verbose=True)

outputs = outputs.cpu().numpy().T

for i, output in enumerate(outputs):
    name = f'undertale-output-{i:03d}.mid'
    encoded_vals_to_midi_file(output, name)
