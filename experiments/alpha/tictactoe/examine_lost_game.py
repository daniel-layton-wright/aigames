import argparse
from aigames.training_manager.alpha_training_manager_lightning import AlphaTrainingRunLightning
from aigames.utils.listeners import GameHistoryListener, RewardListener
from aigames.agent import MinimaxAgent
from aigames.game.tictactoe import FastTicTacToe
from experiments.alphatoe import TicTacToeNetwork, FastTicTacToeEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chk_path', '-c', type=str, help='Path for model checkpoint')
    parser.add_argument('--debug', '-d', action='store_true', help='Enter debug mode after playing')
    args = parser.parse_args()

    # Load the model, setup the listener, minimax agent, game
    try:
        model = AlphaTrainingRunLightning.load_from_checkpoint(args.chk_path)
    except TypeError:
        network = TicTacToeNetwork()
        evaluator = FastTicTacToeEvaluator(network)
        model = AlphaTrainingRunLightning.load_from_checkpoint(args.chk_path, alpha_evaluator=evaluator, game_class=FastTicTacToe)

    print(model.network.state_dict()['base.4.running_mean'])
    print(model.alpha_evaluator.network.state_dict()['base.4.running_mean'])

    history_listener = GameHistoryListener()
    reward_listener = RewardListener(1, 1)
    minimax_agent = MinimaxAgent(FastTicTacToe)

    model.agents[-1].eval()
    game = FastTicTacToe([minimax_agent, model.agents[-1]], [history_listener, reward_listener])

    # play quick tournament to get average reward
    n_games = 100
    total_reward = 0
    for _ in range(n_games):
        game.play()
        total_reward += reward_listener.reward

    avg_reward = total_reward / n_games
    print(f'Average reward: {avg_reward}')

    # play until the model loses
    n_games = 0
    while True:
        game.play()
        n_games += 1
        if game.state.rewards[1] == -1:  # model lost
            break

    print(f'It took {n_games} games for the model to lose.')
    print(history_listener)

    # The last state is the losing state, the second to last is after model's move, so third to last state was crucial
    crucial_state = history_listener.history[-3]

    print('Crucial state: ')
    print(crucial_state)

    print('Network evaluation: ')
    print(model.alpha_evaluator.evaluate(crucial_state))

    if args.debug:
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    main()
