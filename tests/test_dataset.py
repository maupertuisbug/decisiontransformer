import torch 
from dataset import reward_to_return, DatasetGenerator


class TestDataset:

    def test_reward_to_return(self):

        rewards = torch.tensor([[1, 2, 3, 4, 5], [2, 4, 2, 4, 2]], dtype=torch.double)

        result  = reward_to_return(rewards, 5)

        expected = torch.tensor([[15, 14, 12, 9, 5],
                                [14, 12, 8, 6, 2]], dtype = torch.double)

        assert torch.allclose(result, expected)


    def test_setup(self):

        dataset_id = 'mujoco/halfcheetah/medium-v0'

        dg = DatasetGenerator(64, 30, dataset_id)
        dg.setup_episodes(10)

        assert len(dg.sampled_episodes) == 10

    
    def test_get_dataset(self):

        dataset_id = 'mujoco/halfcheetah/medium-v0'

        dg = DatasetGenerator(64, 30, dataset_id)
        dg.setup_episodes(10)

        s, a, r, sn, an, rn = dg.get_dataset()

        assert s.shape == (64, 30, 17)