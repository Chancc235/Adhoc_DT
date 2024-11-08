import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from utils import load_data

states, next_states, obs, actions, rewards = load_data("data/PP4a_episodes_data.pt")
data = {
    'state': states,
    'next_state': next_states,
    'obs': obs,
    'action': actions,
    'reward': rewards
}
torch.save(data, "data/PP4a_transition_data.pt")
print(states.shape)
print(obs.shape)