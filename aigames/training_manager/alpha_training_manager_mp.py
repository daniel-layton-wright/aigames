from ctypes import c_int
import time
from tqdm.auto import tqdm
from .alpha_training_manager_lightning import AlphaTrainingRunLightning, AlphaTrainingHyperparametersLightning, \
    BasicAlphaDatasetLightning
import torch.multiprocessing as mp
import queue
import torch.utils.data
from .training_manager import ListDataset


class AlphaTrainingMPData:
    def __init__(self):
        self.stop = mp.Value(c_int)  # increment this to progressively stop processes
        self.n_games = mp.Value(c_int)


class AlphaEvaluatorMP:
    def __init__(self, model: AlphaTrainingRunLightning, n_procs=1):
        self.model = model
        self.queue = mp.Queue()
        self.response_queues = [mp.Queue() for _ in range(n_procs)]
        self.response_queue_i = 0  # This needs to be overwritten in the subprocess

    def evaluation_loop(self, mp_data: AlphaTrainingMPData):
        # Loop until we get a stop signals
        while mp_data.stop.value < 2:
            try:
                data = self.queue.get(block=False)
            except queue.Empty:
                continue

            # Evaluate the state and send it back
            processed_state, i = data
            result = self.model(processed_state)
            self.response_queues[i].put(result)

    def evaluate(self, state):
        # Send the state to the queue
        processed_state = self.model.alpha_evaluator.process_state(state)
        data = (processed_state, self.response_queue_i)
        self.queue.put(data)

        # Wait for response and return
        result = self.response_queues[self.response_queue_i].get()
        return result


class AlphaSelfPlayMP:
    def __init__(self, game):
        self.game = game

    def self_play_loop(self, response_queue_index: int, mp_data: AlphaTrainingMPData):
        self.game.players[0].evaluator.response_queue_i = response_queue_index

        # Play at least one game
        self.game.play()
        mp_data.n_games.value += 1

        # Loop until we get a stop signal, playing the game
        while mp_data.stop.value < 1:
            self.game.play()
            mp_data.n_games.value += 1


class AlphaTrainingHyperparametersLightningMP(AlphaTrainingHyperparametersLightning):
    __slots__ = ['n_self_play_procs', 'n_self_play_games', 'self_play_every_n_epochs', 'n_data_loader_workers']

    def __init__(self):
        super().__init__()
        self.n_self_play_procs = 1  # Number of processes to use for self-play
        self.n_self_play_games = 1000  # Number of games to play in each self-play run
        self.self_play_every_n_epochs = 4  # Do self-play every n epochs
        self.n_data_loader_workers = 1


class BasicAlphaDatasetLightningMP(BasicAlphaDatasetLightning):
    def __init__(self, evaluator, hyperparams: AlphaTrainingHyperparametersLightningMP, process_state=True):
        super().__init__(evaluator, hyperparams, process_state)
        self.queue_for_data_to_add = mp.Queue()  # Queue for data to add to the dataset
        self.hyperparams = hyperparams

    def __iter__(self):
        dataset = ListDataset(self.states, self.pis, self.rewards)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.hyperparams.batch_size, shuffle=True,
                                                 num_workers=self.hyperparams.n_data_loader_workers)
        return iter(dataloader)

    def on_data_point(self, state, pi, reward):
        self.queue_for_data_to_add.put((state, pi, reward))

    def add_data_from_queue(self):
        while True:
            try:
                state, pi, reward = self.queue_for_data_to_add.get(block=False)
            except queue.Empty:
                return
            else:
                super().on_data_point(state, pi, reward)


class AlphaTrainingRunLightningMP(AlphaTrainingRunLightning):
    def __init__(self, game_class, network, alpha_evaluator, hyperparams: AlphaTrainingHyperparametersLightningMP):
        super().__init__(game_class, network, alpha_evaluator, hyperparams)
        self.hyperparams = hyperparams  # Just to shut PyCharm up

        # Update the evaluator to use the multiprocessing version
        self.alpha_evaluator_mp = AlphaEvaluatorMP(self, n_procs=hyperparams.n_self_play_procs)
        self.agent.alpha_evaluator = self.alpha_evaluator_mp

        # Update the dataset to use the multiprocessing version
        self.agent.listeners.remove(self.dataset)
        self.dataset = BasicAlphaDatasetLightningMP(self.alpha_evaluator, hyperparams)
        self.agent.listeners.append(self.dataset)

    def on_fit_start(self):
        # Do first self-play
        self.mp_self_play(self.hyperparams.n_self_play_games)

    def on_train_batch_start(self, batch, batch_idx: int):
        pass  # In the parent class, it does self play here

    def on_train_start(self):
        pass  # In the parent class, it does self play here

    def on_train_epoch_start(self) -> None:
        self.network.train()

    def on_train_epoch_end(self) -> None:
        # Do self-play games if it's time
        if (self.current_epoch + 1) % self.hyperparams.self_play_every_n_epochs == 0:
            self.mp_self_play(self.hyperparams.n_self_play_games)

    def mp_self_play(self, n_games):
        # This makes sure the agent collects data:
        self.agent.train()

        # When doing self-play, we are not training the network, only generating training data
        # And we need the network in eval mode for, e.g., batch-norm
        self.network.eval()

        mp_data = AlphaTrainingMPData()
        self_play_processes = []

        eval_proc = mp.Process(target=self.alpha_evaluator_mp.evaluation_loop, args=(mp_data,))
        eval_proc.start()

        self_play_mp = AlphaSelfPlayMP(self.game)

        # Make sure we don't start more processes than necessary
        n_self_play_procs = min(self.hyperparams.n_self_play_procs, n_games)
        for i in range(n_self_play_procs):
            proc = mp.Process(target=self_play_mp.self_play_loop, args=(i, mp_data))
            proc.start()
            self_play_processes.append(proc)

        # Loop and count the number of games, updating tqdm progress bar
        with tqdm(total=n_games) as pbar:
            # Account for the n_self_play procs that will finish
            while (mp_data.n_games.value + n_self_play_procs) < n_games:
                pbar.update(mp_data.n_games.value - pbar.n)

                # In the main process we need to add the data that is coming in from the self-play to the dataset
                self.dataset.add_data_from_queue()

            pbar.update(mp_data.n_games.value - pbar.n)

            # Stop the processes, wait for the self-play processes to finish first, then kill the eval process
            mp_data.stop.value += 1
            while len(self_play_processes) > 0:
                proc = self_play_processes[0]
                proc.join(timeout=1)
                pbar.update(mp_data.n_games.value - pbar.n)
                if proc.exitcode is None:  # The process is still running
                    self.dataset.add_data_from_queue()  # Need to empty the data queue to be able to join process
                else:  # The process finished
                    self_play_processes.remove(proc)

        self.dataset.add_data_from_queue()

        mp_data.stop.value += 1
        eval_proc.join()
