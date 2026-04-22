import torch 
from transformer_heads import FeedForward, SingleHead
from dataset import DatasetGenerator
import gymnasium as gym
from tqdm import tqdm
import numpy as np
torch.set_default_dtype(torch.float64)
params = {
    'block_size': 20, 
    'n_embed' : 512, 
    'state_n' : 17,
    'action_n' : 6
}

class DecisionTransformer(torch.nn.Module):

    def __init__(self, params):

        super().__init__()

        self.horizon_length = params['block_size']
        self.n_embed  = int(params['n_embed'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_n = params['state_n']
        action_n = params['action_n']

        self.state_embedding = torch.nn.Linear(state_n, self.n_embed).to(self.device)
        self.action_embedding = torch.nn.Linear(action_n, self.n_embed).to(self.device)
        self.reward_embedding = torch.nn.Linear(1, self.n_embed).to(self.device)

        self.pos_embedding = torch.nn.Embedding(self.horizon_length, self.n_embed).to(self.device)
        self.embed_ln = torch.nn.LayerNorm(self.n_embed).to(self.device)

        self.predict_state = torch.nn.Linear(self.n_embed, state_n).to(self.device)
        self.predict_action = torch.nn.Sequential(
            torch.nn.Linear(self.n_embed, action_n), 
            torch.nn.Tanh()
        ).to(self.device)
        self.predict_return = torch.nn.Linear(self.n_embed, 1).to(self.device)

        self.attn_head = SingleHead(self.n_embed, self.n_embed, 3*self.horizon_length).to(self.device)
        self.ffn       = FeedForward(self.n_embed).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0003)

        self.dg = DatasetGenerator(64, self.horizon_length, 'mujoco/halfcheetah/medium-v0')
        self.dg.setup_episodes(100)
        

    def forward(self, states, actions, returns_to_go, horizon_length):

        pos_em = self.pos_embedding(torch.arange(0, horizon_length, device = self.device))
        state_em = self.state_embedding(states) + pos_em
        action_em = self.action_embedding(actions) + pos_em
        reward_em   = self.reward_embedding(returns_to_go) + pos_em

        batch_size, horizon_length, n_embed = state_em.shape
        stack_input = torch.stack([state_em, action_em, reward_em], dim=2).reshape(batch_size, 3*horizon_length, n_embed)   # batch_size, horizon, return+state+action - nembed
        stack_input = self.embed_ln(stack_input)

        output = self.attn_head(stack_input)
        output = self.ffn(output)  # batch_size, state, action, return

        out = output.reshape(batch_size, horizon_length, 3, self.n_embed).permute(0, 2, 1, 3)

        state_preds  = self.predict_state(out[:, 0])
        action_preds = self.predict_action(out[:, 1])
        return_preds = self.predict_return(out[:, 2])

        return output, state_preds, action_preds, return_preds


    def learn(self):

        s, a, r, sn, an, rn = self.dg.get_dataset()

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        sn = sn.to(self.device)
        an = an.to(self.device)
        rn = rn.to(self.device)

        _, _, action_preds, _ = self(s, a, r, 20)

        loss_fn = torch.nn.MSELoss()

        action_actual = an[:, -1, :]
        action_pred   = action_preds[:, -1, :]

        output = loss_fn(action_actual, action_pred)
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

    def eval(self):

        env = gym.make('HalfCheetah-v5')
        rl = []

        for ep in range(0, 10):
            obs, _ = env.reset()
            action = env.action_space.sample()
            next_obs, reward_ep, _, _, _ = env.step(action)
            states = torch.tensor(obs, device = self.device, dtype = torch.float64).unsqueeze(0).repeat(self.horizon_length, 1).unsqueeze(0)
            actions = torch.tensor(action, device = self.device, dtype = torch.float64).unsqueeze(0).repeat(self.horizon_length, 1).unsqueeze(0)
            returns = torch.tensor(5000.0, device = self.device, dtype = torch.float64).unsqueeze(0).repeat(self.horizon_length, 1).unsqueeze(0)
            return_ = 5000.0
            steps = 0
            while steps < 1000:

                _, _, action_preds, _ = self(states, actions, returns, 20)
                action = action_preds[:, -1, :].squeeze(0).detach().cpu().numpy()
                next_obs, reward, _, _, _ = env.step(action)
                return_ = return_ - reward

                states = (torch.cat([states.squeeze(0), torch.tensor(next_obs, device=self.device, dtype = torch.float64).unsqueeze(0)], dim=0)[1:,:]).unsqueeze(0)
                actions = (torch.cat([actions.squeeze(0), torch.tensor(action, device=self.device, dtype = torch.float64).unsqueeze(0)], dim=0)[1:,:]).unsqueeze(0)
                returns = (torch.cat([returns.squeeze(0), torch.tensor(return_, device=self.device, dtype = torch.float64).unsqueeze(0).unsqueeze(0)], dim=0)[1:,:]).unsqueeze(0)
                returns = (torch.cat([returns.squeeze(0), torch.tensor(return_, device=self.device, dtype = torch.float64).unsqueeze(0).unsqueeze(0)], dim=0)[1:,:]).unsqueeze(0)

                reward_ep = reward_ep + reward
                steps+=1

            rl.append(reward_ep)

        return np.mean(rl)











dt = DecisionTransformer(params)

for i in tqdm(range(0, 50000)):
    dt.learn()

    if i%500 == 0:
        result = dt.eval()
        print(result)












