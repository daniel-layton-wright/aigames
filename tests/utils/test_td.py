import unittest
from aigames.utils.td import compute_td_targets, compute_td_targets_truncated_slow, compute_td_targets_truncated
import torch


class TestTd(unittest.TestCase):
    def test_simple_td(self):
        state_values = torch.tensor([0., 0., 0., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])

        expected_td_values = torch.tensor([4., 3., 2., 1.])
        td_values = compute_td_targets(state_values, rewards, td_lambda=1.0, discount=1.0)
        torch.testing.assert_close(td_values, expected_td_values)

        expected_td_values = torch.tensor([1.875, 1.75, 1.5, 1.])
        td_values = compute_td_targets(state_values, rewards, td_lambda=1.0, discount=0.5)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td(self):
        state_values = torch.tensor([0., 0., 0., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])

        td_values = compute_td_targets_truncated_slow(state_values, rewards, td_lambda=1.0, discount=1.0, truncate_length=1)
        expected_td_values = torch.tensor(1.)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td2(self):
        state_values = torch.tensor([0., 1., 0., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])
        td_values = compute_td_targets_truncated_slow(state_values, rewards, td_lambda=1.0, discount=1.0, truncate_length=1)
        expected_td_values = torch.tensor(2.)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td3(self):
        state_values = torch.tensor([0., 0., 1., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])
        td_values = compute_td_targets_truncated_slow(state_values, rewards, td_lambda=1.0, discount=1.0, truncate_length=2)
        expected_td_values = torch.tensor(3.)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td4(self):
        state_values = torch.tensor([0., 0., 1., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])
        td_values = compute_td_targets_truncated_slow(state_values, rewards, td_lambda=0.5, discount=1.0, truncate_length=2)
        expected_td_values = torch.tensor(2.)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td5(self):
        state_values = torch.tensor([0., 0., 1., 0.])
        rewards = torch.tensor([1., 1., 1., 1.])
        td_values = compute_td_targets_truncated(state_values, rewards, td_lambda=0.5, discount=1.0, truncate_length=2)
        expected_td_values = torch.tensor(2.)
        torch.testing.assert_close(td_values, expected_td_values)

    def test_trunctated_td6(self):
        state_values = torch.tensor([0.])
        rewards = torch.tensor([12.])
        td_values = compute_td_targets_truncated(state_values, rewards, td_lambda=0.5, discount=1.0, truncate_length=5)
        expected_td_values = torch.tensor(12.)
        torch.testing.assert_close(td_values, expected_td_values)
