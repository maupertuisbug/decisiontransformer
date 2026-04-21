import torch 
from transformer_heads import FeedForward, SingleHead
params = {
    'block_size': 30, 
    'n_embed' : 512
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
        

    def forward(self, states, actions, returns_to_go, horizon_length):

        # batch, horizon_length, state_n

        pos_em = self.pos_embedding(torch.arange(0, horizon_length, device = self.device))
        state_em = self.state_embedding(states) + pos_em
        action_em = self.action_embedding(actions) + pos_em
        reward_em   = self.reward_embedding(returns_to_go) + pos_em

        stack_input = torch.stack([state_em, action_em, reward_em], dim=2)   # batch_size, horizon, return+state+action - nembed

        stack_input = self.embed_ln(stack_input)

        output = self.attn_head(stack_input)
        output = self.ffn(output)  # batch_size, state, action, return

        output = output.reshape(batch_size, horizon_length, 3, self.n_embed).permute(0, 2, 1, 3)

        state_preds  = self.predict_state(x[:, 0])
        action_preds = self.predict_action(x[:, 1])
        reward_preds = self.predict_reward(x[:, 2])

        return output, state_preds, action_preds, return_preds








