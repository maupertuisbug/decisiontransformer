import minari 
import torch 
import random
from torch import vmap

def reward_to_return(rewards, length):
    mask = torch.tril(torch.ones(length, length, dtype=torch.double))
    rewards = rewards @ mask
    return rewards

class DatasetGenerator:

    def __init__(self, batch_size, horizon, dataset_id):
        self.batch_size = batch_size
        self.horizon = horizon
        self.dataset_id = dataset_id
        self.dataset = minari.load_dataset(self.dataset_id, download=True)

    def setup_episodes(self, n_ep):

        self.sampled_episodes = self.dataset.sample_episodes(n_ep)
        self.ep_length = self.sampled_episodes[0].observations.shape[0]

    def get_dataset(self):

        idx = torch.randint(self.ep_length-self.horizon-1, (self.batch_size,))

        random_episode = random.randint(0, 9)

        states = self.sampled_episodes[random_episode].observations
        actions = self.sampled_episodes[random_episode].actions
        rewards = self.sampled_episodes[random_episode].rewards

        data_states = torch.stack([torch.tensor(states[x:x+self.horizon], dtype = torch.float64) for x in idx])
        data_actions = torch.stack([torch.tensor(actions[x:x+self.horizon], dtype = torch.float64) for x in idx])
        data_rewards = torch.stack([torch.tensor(rewards[x:x+self.horizon], dtype = torch.float64) for x in idx])

        data_rewards = reward_to_return(data_rewards, self.horizon).unsqueeze(-1)

        data_states_next = torch.stack([torch.tensor(states[x+1:x+1+self.horizon], dtype = torch.float64) for x in idx])
        data_actions_next = torch.stack([torch.tensor(actions[x+1:x+1+self.horizon], dtype = torch.float64) for x in idx])
        data_rewards_next = torch.stack([torch.tensor(rewards[x+1:x+1+self.horizon], dtype = torch.float64) for x in idx])

        data_rewards_next = reward_to_return(data_rewards_next, self.horizon).unsqueeze(-1)

        return data_states, data_actions, data_rewards, data_states_next, data_actions_next, data_rewards_next




