import unittest
from aigames.experiments.alpha.G2048Multi.network_architectures import G2048MultiNetworkV2
import torch


class TestG2048NetworkV2(unittest.TestCase):
    def test_scale_value(self):

        network = G2048MultiNetworkV2()
        value = torch.tensor([0, 1, 2, 3, 4, 5, -123, 23434, -394, 12345108, 1000, 2000], dtype=torch.float32)
        torch.testing.assert_close(value, network.inverse_scale(network.scale_value(value)), atol=0.035, rtol=0.001)

    def test_max_in_corner(self):
        network = G2048MultiNetworkV2()

        states = torch.FloatTensor([[[1, 0, 1, 0],
                                     [0, 0, 0, 0],
                                     [2, 0, 3, 0],
                                     [4, 4, 4, 3]],
                                    [[0, 0, 1, 1],
                                     [1, 1, 1, 1],
                                     [2, 0, 0, 2],
                                     [0, 0, 3, 2]],
                                    [[0, 0, 1, 9],
                                     [1, 1, 1, 1],
                                     [2, 0, 0, 2],
                                     [8, 0, 3, 2]],
                                    [[9, 0, 1, 9],
                                     [1, 1, 1, 1],
                                     [2, 10, 0, 2],
                                     [9, 0, 3, 9]]])

        processed_states = network.process_state(states)

        kernel = -4 * torch.ones(15, 16, 4, 4)
        for i in range(1, 16):
            kernel[i-1, :(i+1)] = 0
            kernel[i-1, i, 0, 0] = 1
            kernel[i-1, i, 0, 3] = 1
            kernel[i-1, i, 3, 0] = 1
            kernel[i-1, i, 3, 3] = 1

        conv = torch.nn.functional.conv2d(processed_states, kernel, padding=0, stride=1)
        conv = conv.amax(dim=(1, 2, 3))
        conv = (conv > 0)
        torch.testing.assert_close(conv, torch.BoolTensor([True, False, True, False]))
