import unittest

import torch.testing

from aigames.experiments.alpha.hearts.network_architectures import HeartsNetworkHyperparameters, HeartsNetwork
from aigames.game.hearts import Hearts


class TestHeartsNetwork(unittest.TestCase):
    def setUp(self):
        self.network = HeartsNetwork(HeartsNetworkHyperparameters())

    def test_evaluate_works(self):
        # Test that the evaluate method works
        state = Hearts.get_initial_states(10)
        pi, v = self.network.evaluate(state)
        self.assertEqual(pi.shape, (10, 52))
        self.assertEqual(v.shape, (10, 4))

        # The pi values should sum to 1 and all be close to 1/52
        torch.testing.assert_close(pi.sum(dim=-1), torch.ones(10))
        torch.testing.assert_close(pi, 1/52 * torch.ones_like(pi), rtol=0.05, atol=0.01)

        # The values should be close to -13
        torch.testing.assert_close(v, -13 * torch.ones_like(v), rtol=0.05, atol=0.01)
