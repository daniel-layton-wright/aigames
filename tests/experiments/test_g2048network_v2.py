import unittest


class TestG2048NetworkV2(unittest.TestCase):
    def test_scale_value(self):
        from aigames.experiments.alpha.G2048Multi.network_architectures import G2048MultiNetworkV2
        import torch

        network = G2048MultiNetworkV2()
        value = torch.tensor([0, 1, 2, 3, 4, 5, -123, 23434, -394, 12345108, 1000, 2000], dtype=torch.float32)
        torch.testing.assert_close(value, network.inverse_scale(network.scale_value(value)), atol=0.035, rtol=0.001)
