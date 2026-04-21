import torch 






class FeedForward(torch.nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_embed, 4*n_embed), 
                         torch.nn.ReLU(),
                         torch.nn.Linear(4*n_embed, n_embed))

    def forward(self, x):

        out = self.net(x)
        return out

class SingleHead(torch.nn.Module):

    def __init__(self, n_embed_input, n_embed_out, t):

        super().__init__()

        self.key = torch.nn.Linear(n_embed_input, n_embed_out, bias = False)
        self.query = torch.nn.Linear(n_embed_input, n_embed_out, bias = False)
        self.value = torch.nn.Linear(n_embed_input, n_embed_out, bias = False)

        self.t = t 
        self.register_buffer('tril', torch.tril(torch.ones(t, t)))

    def forward(self, x):

        k = self.key(x)
        q = self.query(x)

        aff = q @ k.transpose(-2, -1)

        aff = aff.masked_fill((self.tril[:self.t, :self.t] == 0), value = float(-inf))
        aff = torch.nn.functional.softmax(aff, dim=-1)

        out = aff @ self.value(x)

        out = self.layer(out)

        return out