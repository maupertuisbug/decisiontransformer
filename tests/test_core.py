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

        states = torch.randn((32, 30, 17)).to('cuda')
        actions = torch.randn((32, 30,  8)).to('cuda')
        returns_to_go =  torch.randn((32, 30, 1)).to('cuda')

        output,state_p, action_p, return_p = dt.forward(states, actions, returns_to_go, 30)

        assert state_p.shape == (32, 30, 17)
        assert action_p.shape == (32, 30, 8)
        assert return_p.shape == (32, 30, 1)

    def test_init_state(self):

        state = torch.tensor([1, 2, 3])
        state = state.unsqueeze(0)

        states = state.repeat(3,1)

        expected_states = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        assert torch.allclose(states, expected_states)





