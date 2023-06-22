from ctypes import c_int
from tqdm.auto import tqdm
from .alpha_training_manager_lightning import AlphaTrainingRunLightning, AlphaTrainingHyperparametersLightning, \
    BasicAlphaDatasetLightning
import multiprocessing as mp
import queue


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

        # Loop until we get a stop signal, playing the game
        while mp_data.stop.value < 1:
            self.game.play()
            mp_data.n_games.value += 1


class BasicAlphaDatasetLightningMP(BasicAlphaDatasetLightning):
    def __init__(self, evaluator, hyperparams, process_state=True):
        super().__init__(evaluator, hyperparams, process_state)
        self.queue_for_data_to_add = mp.Queue()  # Queue for data to add to the dataset

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


class AlphaTrainingHyperparametersLightningMP(AlphaTrainingHyperparametersLightning):
    __slots__ = ['n_self_play_procs']

    def __init__(self):
        super().__init__()
        self.n_self_play_procs = 1


class AlphaTrainingRunLightningMP(AlphaTrainingRunLightning):
    def __init__(self, game_class, alpha_evaluator, hyperparams: AlphaTrainingHyperparametersLightningMP):
        super().__init__(game_class, alpha_evaluator, hyperparams)

        # Update the evaluator to use the multiprocessing version
        self.alpha_evaluator_mp = AlphaEvaluatorMP(self, n_procs=hyperparams.n_self_play_procs)
        self.agent.alpha_evaluator = self.alpha_evaluator_mp

        # Update the dataset to use the multiprocessing version
        self.agent.listeners.remove(self.dataset)
        self.dataset = BasicAlphaDatasetLightningMP(self.alpha_evaluator, hyperparams)
        self.agent.listeners.append(self.dataset)

    def mp_self_play(self, n_games):
        self.network.eval()  # When doing self-play, we are not training the network, only generating training data

        mp_data = AlphaTrainingMPData()
        self_play_processes = []

        eval_proc = mp.Process(target=self.alpha_evaluator_mp.evaluation_loop, args=(mp_data,))
        eval_proc.start()

        self_play_mp = AlphaSelfPlayMP(self.game)

        for i in range(self.hyperparams.n_self_play_procs):
            proc = mp.Process(target=self_play_mp.self_play_loop, args=(i, mp_data))
            proc.start()
            self_play_processes.append(proc)

        # Loop and count the number of games, updating tqdm progress bar
        with tqdm(total=n_games) as pbar:
            while mp_data.n_games.value < n_games:
                pbar.update(mp_data.n_games.value - pbar.n)

                # In the main process we need to add the data that is coming in from the self-play to the dataset
                self.dataset.add_data_from_queue()

        # Stop the processes, wait for the self-play processes to finish first, then kill the eval process
        mp_data.stop.value += 1
        while len(self_play_processes) > 0:
            proc = self_play_processes[0]
            try:
                proc.join(timeout=1)
            except mp.TimeoutError:
                self.dataset.add_data_from_queue()  # Need to empty the data queue to be able to join process
            else:
                self_play_processes.remove(proc)

        mp_data.stop.value += 1
        eval_proc.join()
