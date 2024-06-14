from collections import defaultdict
from typing import List, Tuple

from aigames.agent.alpha_agent_multi import TrainingTau


class TrainingTauDecreaseOnPlateau(TrainingTau):
    def __init__(self, tau_schedule: List[float], plateau_metric, plateau_patience,
                 max_optimizer_steps_before_tau_decrease: int = -1):
        super().__init__(0)
        self.tau_schedule = tau_schedule
        self.i = 0
        self.metrics = defaultdict(list)
        self.plateau_metric = plateau_metric
        self.plateau_patience = plateau_patience
        self.j = -1
        self.max_j = -1
        self.max_metric = None
        self.max_optimizer_steps_before_tau_decrease = max_optimizer_steps_before_tau_decrease
        self.last_self_optimizer_step_tau_decrease = 0
        self.optimizer_step = 0

    def get_tau(self, move_number):
        return self.tau_schedule[self.i]

    def backwards_compatible_check(self):
        if not hasattr(self, 'optimizer_step'):
            self.optimizer_step = 0
            self.last_self_optimizer_step_tau_decrease = 0
            self.max_optimizer_steps_before_tau_decrease = -1

    def update_metric(self, key, val):
        self.backwards_compatible_check()

        if key == 'optimizer_step':
            self.optimizer_step = val

            if 0 < self.max_optimizer_steps_before_tau_decrease <= val - self.last_self_optimizer_step_tau_decrease:
                self.i = min(self.i + 1, len(self.tau_schedule) - 1)
                self.max_metric = None
                self.max_j = self.j
                self.last_self_optimizer_step_tau_decrease = val

        if key != self.plateau_metric:
            return

        self.j += 1

        if self.max_metric is None or val > self.max_metric:
            self.max_metric = val
            self.max_j = self.j

        if self.j - self.max_j >= self.plateau_patience:
            self.i = min(self.i + 1, len(self.tau_schedule) - 1)
            self.max_metric = None
            self.max_j = self.j
            self.last_self_optimizer_step_tau_decrease = self.optimizer_step


class TrainingTauStepSchedule(TrainingTau):
    """
    A training tau based on a schedule for the optimizer step number

    Example: TrainingTauStepSchedule([(1.0, int(100e3)), (0.5, int(200e3)), (0.1, int(300e3)), (0.0, None)])
    """
    def __init__(self, schedule: List[Tuple[float, int]]):
        super().__init__(0)
        self.schedule = schedule
        self.i = 0

    def update_metric(self, key, val):
        if key == 'optimizer_step':
            self.i = next((i for i, (_, step) in enumerate(self.schedule) if (step is None or step > val)),
                          len(self.schedule) - 1  # default
                          )

    def get_tau(self, move_number):
        return self.schedule[self.i][0]
