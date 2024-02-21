from aigames import CommandLineGame
from ....agent.alpha_agent_multi import BaseAlphaEvaluator, AlphaAgentMulti, AlphaAgentHyperparametersMulti
import torch
import torch.nn.functional as F
from ....game.G2048_multi import G2048Multi
from ....utils.listeners import ActionCounterProgressBar
from ....utils.utils import add_all_slots_to_arg_parser, load_from_arg_parser


class CustomG2048Evaluator(BaseAlphaEvaluator):
    def __init__(self,):
        super().__init__()
        self.zero_bonus = 1000
        self.max_in_corner_bonus = 1000
        self.bonus_for_tiles_one_away = 20

    def process_state(self, state):
        return state

    def evaluate(self, states):
        pis = torch.tensor([0.4, 0.2, 0.3, 0.1], dtype=torch.float32, device=states.device)
        pis = pis.repeat(states.shape[0], 1)

        scores = torch.zeros((states.shape[0],), dtype=torch.float32, device=states.device)
        num_zeros = (states == 0).sum(dim=(1, 2))
        scores += num_zeros * self.zero_bonus

        # Check for max tile in corner
        max_tile = states.amax(dim=(1, 2))
        max_tile_in_corner = torch.zeros((states.shape[0],), dtype=torch.bool, device=states.device)
        max_tile_in_corner[states[:, 0, 0] == max_tile] = True
        max_tile_in_corner[states[:, 0, -1] == max_tile] = True
        max_tile_in_corner[states[:, -1, 0] == max_tile] = True
        max_tile_in_corner[states[:, -1, -1] == max_tile] = True
        scores += max_tile_in_corner * self.max_in_corner_bonus

        # Check for tiles one away from each other
        tiles_one_away = torch.zeros((states.shape[0],), dtype=torch.bool, device=states.device)

        padded_states = F.pad(states, (1, 1, 1, 1), 'constant', -1)
        # Replace zeros with -2 -- no bonus for any zero tiles
        padded_states[padded_states == 0] = -1

        for roll_amt, roll_dim in [(1, -1), (-1, -1), (1, -2), (-1, -2)]:
            one_way = ((padded_states == padded_states.roll(roll_amt, roll_dim)) == 1).sum(dim=(1,2))
            scores += one_way * self.bonus_for_tiles_one_away

        return pis, scores.reshape(-1, 1)


def main():
    hypers = AlphaAgentHyperparametersMulti()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--agent_eval_mode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--show_game', action=argparse.BooleanOptionalAction, default=False)

    add_all_slots_to_arg_parser(parser, hypers)
    args = parser.parse_args()

    load_from_arg_parser(args, hypers)

    evaluator = CustomG2048Evaluator()

    agent = AlphaAgentMulti(G2048Multi, evaluator, hypers)
    game = G2048Multi(args.n_games, agent, listeners=[ActionCounterProgressBar(1000)])

    if args.agent_eval_mode:
        agent.eval()

    if args.show_game:
        game.listeners.append(CommandLineGame(0))

    game.play()

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()


"""
First version of a simple evalutor with 100 mcts iters in 1000 games:
Average best tile: 9.243
Fraction reached 1024: 0.373
Fraction reached 2048: 0.015
"""
