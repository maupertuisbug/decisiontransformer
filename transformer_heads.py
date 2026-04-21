import torch 






class FeedForward(torch.nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(self.n_embed, 4*self.n_embed), 
                         torch.nn.ReLU(),
                         torch.nn.Linear(4*self.n_embed, self.n_embed))

    def forward(self, x):

        out = self.net(x)
        return out