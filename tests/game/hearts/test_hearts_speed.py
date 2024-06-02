import torch

from aigames import GameListenerMulti
from aigames.game.hearts import Hearts, get_legal_action_masks_core, get_next_states_core
import time
from aigames.utils.listeners import ActionCounterProgressBar


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    def duration_in_seconds(self):
        return self.end_time - self.start_time


def get_random_states(n_games):
    # Play Hearts games with a random agent and record all the intermediate states, so that we have a bunch of
    # random but legal states to test speed with
    from aigames.agent.random_agent_multi import RandomAgentMulti
    import torch

    class StateCollector(GameListenerMulti):
        def __init__(self):
            self.states = []
            self.actions = []

        def on_action(self, game, action):
            self.states.append(game.states.clone())
            self.actions.append(action)

    state_collector = StateCollector()
    game = Hearts(n_games, RandomAgentMulti(Hearts), [state_collector, ActionCounterProgressBar(52, 'Generating random states')])
    game.play()

    # Concatenate states, shuffle and return
    return torch.cat(state_collector.states), torch.cat(state_collector.actions)


def test_get_legal_action_masks_speed(states, reps=100):
    # Test the speed of get_legal_action_masks for various sizes of states:
    sizes = [1, 10, 100, 1000, 10000, 100000]
    t = Timer()

    for size in sizes:
        # Compute the indices outside the timer
        indices = [torch.randperm(states.shape[0])[:size] for _ in range(reps)]

        with t:
            for i in range(reps):
                states_size = states[indices[i]]
                get_legal_action_masks_core(states_size)

        # Record the result
        print(f"Size: {size}, Time: {t.duration_in_seconds() / reps:.6f} seconds")

    # Now test the jit version
    for size in sizes:
        # Compute the indices outside the timer
        indices = [torch.randperm(states.shape[0])[:size] for _ in range(reps)]

        with t:
            for i in range(reps):
                states_size = states[indices[i]]
                Hearts.get_legal_action_masks_jit(states_size)

        print(f"Size: {size}, Time (jit): {t.duration_in_seconds() / reps:.6f} seconds")


def test_get_next_states_speed(states, actions, reps=100):
    # Test the speed of get_next_states for various sizes of states:
    sizes = [1, 10, 100, 1000, 10000, 100000]
    t = Timer()

    for size in sizes:
        # Compute the indices outside the timer
        indices = [torch.randperm(states.shape[0])[:size] for _ in range(reps)]

        with t:
            for i in range(reps):
                states_size = states[indices[i]]
                actions_size = actions[indices[i]]
                get_next_states_core(states_size, actions_size)

        # Record the result
        print(f"Size: {size}, Time: {t.duration_in_seconds() / reps:.6f} seconds")

    # Now test the jit version
    for size in sizes:
        # Compute the indices outside the timer
        indices = [torch.randperm(states.shape[0])[:size] for _ in range(reps)]

        with t:
            for i in range(reps):
                states_size = states[indices[i]]
                actions_size = actions[indices[i]]
                Hearts.get_next_states_jit(states_size, actions_size)

        print(f"Size: {size}, Time (jit): {t.duration_in_seconds() / reps:.6f} seconds")


def main():
    states, actions = get_random_states(2000)
    test_get_legal_action_masks_speed(states)
    test_get_next_states_speed(states, actions)


if __name__ == '__main__':
    main()


"""
State that is causing problems to legal actions jit

tensor([[[ 2,  0, 40,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           3,  0,  0,  4,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0,  1,  0,  0,
           0,  1,  0,  4,  0,  0,  0,  4,  3,  0,  0,  0,  0,  0,  2,  0,  0,
           0,  3,  0,  0],
         [ 0,  0,  0,  1, 31,  0,  9,  2, 21, 10, 11, 40,  3, 22,  4, 37, 34,
           0, 15, 33,  0, 16, 39, 14,  0, 13, 36,  0, 38, 30,  5,  0, 29,  7,
          32,  0,  6,  0, 24, 35,  8,  0,  0, 17, 25, 26, 18, 28,  0, 23, 12,
          20,  0, 19, 27],
         [ 0,  0,  0,  3,  2,  0,  1,  4,  1,  2,  3,  1,  1,  2,  2,  2,  4,
           0,  1,  3,  0,  2,  4,  4,  0,  3,  2,  0,  3,  1,  2,  0,  4,  4,
           3,  0,  3,  0,  4,  1,  1,  0,  0,  3,  2,  3,  4,  1,  0,  3,  4,
           2,  0,  1,  4],
         [ 0,  0,  0,  2,  3,  0,  3,  2,  2,  3,  3,  2,  2,  2,  2,  2,  2,
           0,  3,  2,  0,  3,  2,  3,  0,  3,  2,  0,  2,  3,  1,  0,  3,  1,
           3,  0,  1,  0,  2,  2,  1,  0,  0,  1,  4,  4,  1,  4,  0,  2,  3,
           1,  0,  1,  4],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  0]]], dtype=torch.int8)
           
           *** RuntimeError: The following operation failed in the TorchScript interpreter.
Traceback of TorchScript (most recent call last):
/Users/dlwright/Documents/Projects/aigames/aigames/game/hearts.py(222): get_legal_action_masks_core
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/site-packages/torch/jit/_trace.py(859): trace
/Users/dlwright/Documents/Projects/aigames/aigames/game/hearts.py(259): Hearts
/Users/dlwright/Documents/Projects/aigames/aigames/game/hearts.py(236): <module>
<frozen importlib._bootstrap>(241): _call_with_frames_removed
<frozen importlib._bootstrap_external>(883): exec_module
<frozen importlib._bootstrap>(688): _load_unlocked
<frozen importlib._bootstrap>(1006): _find_and_load_unlocked
<frozen importlib._bootstrap>(1027): _find_and_load
/Users/dlwright/Documents/Projects/aigames/tests/game/hearts/test_hearts_speed.py(4): <module>
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/bdb.py(597): run
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/pdb.py(1566): _runmodule
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/pdb.py(1730): main
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/pdb.py(1759): <module>
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/runpy.py(86): _run_code
/Users/dlwright/anaconda3/envs/aigames/lib/python3.10/runpy.py(196): _run_module_as_main
RuntimeError: shape '[52]' is invalid for input of size 0
"""