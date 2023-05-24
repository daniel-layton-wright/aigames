from aigames.training_manager.alpha_training_manager import *
from aigames.game.connect4 import *
from aigames.agent.alpha_agent import *
from aigames.game.game import AbortGameException
from ctypes import c_bool, c_int
from aigames.game import CommandLineGame
from experiments.alpha_connect4 import Connect4Gui
import queue
from aigames import Flatten
import wandb
import signal
import sys


class Connect4Evaluator(AlphaNetworkEvaluator):
    def process_state(self, state: Connect4State):
        s = state.grid
        if abs(s).sum() % 2 == 1:
            s = copy.deepcopy(s)
            s *= -1
        return s


class Connect4Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            Flatten()
        )

        self.base_out_features = 400

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=7),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(in_features=self.base_out_features, out_features=1),
            nn.Tanh()
        )

    def forward(self, processed_state):
        base = self.base(processed_state)
        policy = self.policy_head(base).squeeze()
        value = self.value_head(base)
        return policy, value


class MultiprocessTrainingListener(TrainingListener, GameListener):
    def __init__(self, monitor_queue: mp.Queue, network: nn.Module,  sentinel: mp.Value,
                 checkpoint_network_every_n_iters=1000, checkpoint_path='checkpoint{}.chk'):
        self.monitor_queue = monitor_queue
        self.n_iters = mp.Value(c_int)
        self.n_games = mp.Value(c_int)
        self.checkpoint_network_every_n_iters = checkpoint_network_every_n_iters
        self.network = network
        self.checkpoint_path = checkpoint_path
        self.sentinel = sentinel

    def on_training_step(self, iter: int, loss: float, training_manager, **kwargs):
        self.n_iters.value += 1
        log_dict = {'iter': self.n_iters.value, 'loss': loss}
        self.monitor_queue.put(log_dict)

        if self.n_iters.value % self.checkpoint_network_every_n_iters == 0 and self.n_iters.value > 0:
            torch.save({
                'network_state': self.network.state_dict(),
                'n_iters': self.n_iters.value,
                'n_games': self.n_games.value
           }, self.checkpoint_path.format(self.n_iters.value))

    def before_action(self, game, legal_actions):
        if not self.sentinel.value:
            raise AbortGameException

    def on_game_end(self, game):
        self.n_games.value += 1
        log_dict = {'n_games': self.n_games.value}
        self.monitor_queue.put(log_dict)

    def on_training_end(self):
        self.monitor_queue.close()
        self.monitor_queue.join_thread()

    def on_data_size_change(self, new_data_size):
        self.monitor_queue.put({'data_size': new_data_size})


class WandbMonitor:
    def __init__(self, game_class):
        self.game_class = game_class
        self.wandb_run = None

    def start(self):
        self.wandb_run = wandb.init(project='aigames2', tags=[self.game_class.__name__, 'alpha'], tensorboard=True)

    def process_queue_item(self, item):
        wandb.log(item)

    def on_training_end(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()


def data_process(self_play_data_queue: mp.Queue, train_minibatch_queue: mp.Queue, dataset: BasicAlphaDataset,
                 minibatch_size: int, training_listener: TrainingListener, sentinel: mp.Value):
    self_play_data_queue.cancel_join_thread()
    train_minibatch_queue.cancel_join_thread()
    data_relayer = AlphaDataRelayer(self_play_data_queue, train_minibatch_queue, dataset, minibatch_size)

    last_data_size = 0

    while sentinel.value:
        data_relayer.check_queue_and_callback()
        data_relayer.sample_minibatch_and_add_to_queue()

        cur_data_size = len(dataset)
        if last_data_size != cur_data_size:
            training_listener.on_data_size_change(cur_data_size)
            last_data_size = cur_data_size


def train_process(train_minibatch_queue: mp.Queue, optimizer: AlphaNetworkOptimizer, training_listeners: List[TrainingListener],
                  sentinel: mp.Value):
    train_minibatch_queue.cancel_join_thread()
    while sentinel.value:
        try:
            (processed_states, pis, rewards) = train_minibatch_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        loss = optimizer.take_training_step_processed(processed_states, pis, rewards)
        for listener in training_listeners:
            listener.on_training_step(-1, loss, None)

    for listener in training_listeners:
        listener.on_training_end()


def self_play_process(game_class: Type[SequentialGame], agent, listener_types, listeners: List[GameListener],
                      training_listeners: List[TrainingListener], n_games: int, n_game_counter: mp.Value,
                      sentinel: mp.Value):
    game = game_class([agent, agent], [listener_type() for listener_type in listener_types] + listeners)
    while n_game_counter.value < n_games and sentinel.value:
        n_game_counter.value += 1
        game.play()

    for listener in training_listeners:
        listener.on_training_end()


def monitor_process(monitor, monitor_queue: mp.Queue, sentinel: mp.Value):
    monitor.start()
    while sentinel.value or not monitor_queue.empty():
        if not sentinel.value:
            monitor_queue.close()

        try:
            monitor_item = monitor_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        monitor.process_queue_item(monitor_item)

    monitor_queue.close()
    monitor_queue.join_thread()
    monitor.on_training_end()


def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)


def main():
    mp.set_start_method('spawn')
    signal.signal(signal.SIGTERM, sigterm_handler)

    n_self_play_procs = 3

    network = Connect4Network()
    network.share_memory()
    evaluator = Connect4Evaluator(network)
    optimizer = AlphaNetworkOptimizer(evaluator, lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=1e-5))

    data_queue = mp.Queue()
    train_minibatch_queue = mp.Queue(maxsize=10)
    monitor_queue = mp.Queue()
    sentinel = mp.Value(c_bool)
    sentinel.value = True

    game_class = Connect4
    data_sender = AlphaDataSender(data_queue, evaluator)
    agent = AlphaAgent(game_class, evaluator, [data_sender], use_tqdm=False, n_mcts=1000)
    monitor_listener = MultiprocessTrainingListener(monitor_queue, network, sentinel)
    monitor = WandbMonitor(Connect4)
    dataset = BasicAlphaDataset(process_state=False, min_size=1000)

    data_proc = mp.Process(target=data_process, kwargs=dict(self_play_data_queue=data_queue, train_minibatch_queue=train_minibatch_queue,
                                                            dataset=dataset,
                                                            minibatch_size=128, training_listener=monitor_listener, sentinel=sentinel))
    data_proc.start()

    n_game_counter = mp.Value(c_int)
    self_play_procs = []
    for i in range(n_self_play_procs):
        self_play_proc = mp.Process(target=self_play_process,
                                    kwargs=dict(game_class=Connect4, agent=agent, n_games=1000, n_game_counter=n_game_counter,
                                                listener_types=[], listeners=[monitor_listener],
                                                training_listeners=[monitor_listener], sentinel=sentinel))
        self_play_proc.start()
        self_play_procs.append(self_play_proc)

    train_proc = mp.Process(target=train_process,
                            kwargs=dict(train_minibatch_queue=train_minibatch_queue, optimizer=optimizer, sentinel=sentinel,
                                        training_listeners=[monitor_listener]))
    train_proc.start()

    monitor_proc = mp.Process(target=monitor_process, kwargs=dict(monitor=monitor, monitor_queue=monitor_queue, sentinel=sentinel))
    monitor_proc.start()

    try:
        for self_play_proc in self_play_procs:
            self_play_proc.join()
    except KeyboardInterrupt:
        print('Caught KeyboardInterrupt')
        sentinel.value = False
    finally:
        for self_play_proc in self_play_procs:
            self_play_proc.join()

    sentinel.value = False
    train_proc.join()
    data_proc.join()
    monitor_proc.join()

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
