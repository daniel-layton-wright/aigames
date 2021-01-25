from aigames.training_manager.alpha_training_manager import *
from aigames.game.connect4 import *
from aigames.agent.alpha_agent import *
from ctypes import c_bool
from aigames.game import CommandLineGame
from alpha_connect4 import Connect4Gui


class PrintDataListener(AlphaAgentListener):
    def on_data_point(self, state, pi, reward):
        print(state, pi, reward)


def data_process(queue: mp.Queue, dataset: AlphaAgentListener, sentinel: mp.Value):
    data_relayer = AlphaDataRecipient(queue, [dataset])
    while sentinel.value:
        data_relayer.check_queue_and_callback()


def self_play_process(game_class: Type[SequentialGame], agent, listener_types, n_games: int):
    game = game_class([agent, agent], [listener_type() for listener_type in listener_types])
    for _ in tqdm(range(n_games)):
        game.play()


def main():
    mp.set_start_method('spawn')
    evaluator = DummyAlphaEvaluator(len(Connect4.get_all_actions()))
    data_queue = mp.Queue()
    data_sentinel = mp.Value(c_bool)
    data_sentinel.value = True
    data_sender = AlphaDataSender(data_queue, evaluator)
    agent = AlphaAgent(Connect4, evaluator, [data_sender], use_tqdm=True)

    data_proc = mp.Process(target=data_process, kwargs=dict(queue=data_queue, dataset=PrintDataListener(), sentinel=data_sentinel))
    data_proc.start()

    self_play_proc = mp.Process(target=self_play_process, kwargs=dict(game_class=Connect4, agent=agent, listener_types=[Connect4Gui], n_games=5))
    self_play_proc.start()
    self_play_proc.join()
    data_sentinel.value = False
    data_proc.join()


if __name__ == '__main__':
    main()
