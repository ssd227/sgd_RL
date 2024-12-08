
from torch.utils.data import Dataset


class PongDataset(Dataset):
    def __init__(self, states, actions, advantages):
        self.states = states  
        self.actions = actions
        self.advantages = advantages

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        advantage = self.advantages[idx]
        return state, action, advantage