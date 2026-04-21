import torch 
from core import DecisionTransformer



class TestCore:


    def test_forward(self):

        params = {
                'block_size': 30, 
                'n_embed' : 512,
                'state_n' : 17,
                'action_n' : 8 }

        dt = DecisionTransformer(params)

        states = torch.randn((32, 30, 512, 17)).to('cuda')
        actions = torch.randn((32, 30, 512, 8)).to('cuda')
        returns_to_go =  torch.randn((32, 30, 512, 1)).to('cuda')

        output,_, _, _ = dt.forward(states, actions, returns_to_go, 30)

        print(output.shape)




