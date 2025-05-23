import torch
import numpy as np
import random
from datetime import datetime


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


# Class to compute a mean iteratively
class RunningMean:
    def __init__(self, name: str = ""):
        self.name = name
        self.restart()

    def restart(self):
        self.mean = 0
        self.n = 0

    def update(self, value):
        self.mean = self.mean + (value - self.mean) / (self.n + 1)
        self.n += 1

    def __str__(self):
        return f"{self.mean}"


def get_output_folder(project_name):
    return project_name + "_" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
