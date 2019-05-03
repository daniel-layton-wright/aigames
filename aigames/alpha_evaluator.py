import torch
import queue


class AlphaEvaluator:
    def evaluate(self, state):
        raise NotImplementedError()

    def train(self, states, action_distns, values):
        raise NotImplementedError()


class MultiprocessingAlphaEvaluator(AlphaEvaluator):
    def __init__(self, id, model, evaluation_queue, results_queue, train_queue):
        self.model = model
        self.evaluation_client = MultiprocessingAlphaEvaluationClient(id, model, evaluation_queue, results_queue)
        self.training_client = MultiprocessingAlphaTrainingClient(train_queue)

    def evaluate(self, state):
        processed_state = self.model.process_state(state)
        return self.evaluation_client.evaluate(processed_state)

    def train(self, states, action_distns, values):
        processed_states = tuple(self.model.process_state(state) for state in states)
        processed_states = torch.cat(processed_states)
        action_distns = torch.stack(tuple(action_distns))
        values = torch.stack(tuple(values))
        return self.training_client.train(processed_states, action_distns, values)


class MultiprocessingAlphaEvaluationClient:
    def __init__(self, id, model, evaluation_queue, results_queue):
        self.id = id
        self.model = model
        self.evaluation_queue = evaluation_queue
        self.results_queue = results_queue

    def evaluate(self, processed_state):
        self.evaluation_queue.put((self.id, processed_state))
        pi, v = self.results_queue.get()
        pi_clone = pi.clone()
        v_clone = v.clone()
        del pi
        del v
        return pi_clone, v_clone


class MultiprocessingAlphaEvaluationWorker:
    def __init__(self, model, device, evaluation_queue, results_queues, kill_queue):
        self.model = model
        self.device = device
        self.evaluation_queue = evaluation_queue
        self.results_queues = results_queues
        self.kill_queue = kill_queue

    def evaluate_until_killed(self):
        while True:
            if not self.kill_queue.empty() and self.evaluation_queue.empty():
                break

            try:
                id, processed_state = self.evaluation_queue.get(timeout=1)
            except queue.Empty:
                continue

            with torch.no_grad():
                result = self.model(processed_state.to(self.device))
                self.results_queues[id].put(result)


class MultiprocessingAlphaTrainingClient:
    def __init__(self, train_queue):
        self.train_queue = train_queue

    def train(self, states, action_distns, values):
        self.train_queue.put((states, action_distns, values))


class MultiprocessingAlphaTrainingWorker:
    def __init__(self, model, optimizer, device, train_queue, kill_queue):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_queue = train_queue
        self.kill_queue = kill_queue

    def train_until_killed(self):
        while True:
            if not self.kill_queue.empty() and self.train_queue.empty():
                break

            try:
                processed_states, action_distns, values = self.train_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Run through network
            pred_distns, pred_values = self.model(processed_states)

            # Compute loss
            losses = (values - pred_values) ** 2 - torch.sum(action_distns * torch.log(pred_distns), dim=1)
            loss = torch.sum(losses)

            # Backprop
            loss.backward()
            self.optimizer.step()
