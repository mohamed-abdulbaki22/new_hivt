from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric

# In metrics/ade.py
class FDE(Metric):
    def __init__(self) -> None:  # Remove all parameters
        super().__init__()  # No arguments here
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    # Keep update() and compute() unchanged
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # final displacement error at last timestep
        self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
