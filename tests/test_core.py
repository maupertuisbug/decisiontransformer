import torch 





class TestCore:


    def test_stack(self):

        state = torch.tensor(([[1, 2], [1, 2]], [[1, 2], [1, 2]]))
        action = torch.tensor(([[21, 31], [22,32]], [[23, 33], [24, 34]]))

        print(state.shape, action.shape)
        stack = torch.stack((state, action), dim=1)
        print(stack)

        # assert torch.allclose(stack, torch.tensor([[[ 1,  2, 21, 31],
        #                         [ 1,  2, 22, 32]],
        #                         [[ 1,  2, 23, 33],
        #                         [ 1,  2, 24, 34]]]))
