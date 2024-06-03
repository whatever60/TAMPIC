import torch
import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmupScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        T_max: int,
        T_warmup: int,
        start_lr: float,
        peak_lr: float = None,
        end_lr: float,
        mode: str = "cosine",
    ):
        """
        Initialize the WarmupScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            T_warmup (int): Number of iterations for the warmup phase.
            start_lr (float): Starting learning rate.
            end_lr (float): Ending learning rate.
            mode (str): Mode of the scheduler, either 'cosine' or 'linear'.
        """
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.mode = mode
        self.peak_lr = optimizer.param_groups[0]["lr"] if peak_lr is None else peak_lr
        self.start_lr_multiplier = start_lr / self.peak_lr
        self.end_lr_multiplier = end_lr / self.peak_lr
        super().__init__(optimizer, self._decay_func)

    def _decay_func(self, iteration: int) -> float:
        """
        Calculate the learning rate multiplier.

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: Learning rate multiplier.
        """
        if iteration <= self.T_warmup:
            multiplier = self.start_lr_multiplier + (iteration / self.T_warmup) * (
                1 - self.start_lr_multiplier
            )
        else:
            progress = (iteration - self.T_warmup) / (self.T_max - self.T_warmup)
            if self.mode == "cosine":
                multiplier = self.end_lr_multiplier + (
                    0.5
                    * (1 + math.cos(math.pi * progress))
                    * (1 - self.end_lr_multiplier)
                )
            elif self.mode == "linear":
                multiplier = 1 + progress * (self.end_lr_multiplier - 1)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
        return multiplier


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dummy parameters
    parameters = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    total_iters = 2000
    warmup_iters = 100

    # Test WarmupScheduler with cosine annealing
    optimizer = torch.optim.Adam([parameters], lr=0.2)
    scheduler = WarmupScheduler(
        optimizer,
        T_max=total_iters,
        T_warmup=warmup_iters,
        start_lr=0.02,
        end_lr=0.1,
        mode="cosine",
    )
    actual_lr = []
    for _iter in range(total_iters):
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, label="CosineAnnealingLRWarmup")

    # Test WarmupScheduler with linear warmup
    optimizer = torch.optim.Adam([parameters], lr=0.2)
    scheduler = WarmupScheduler(
        optimizer,
        T_max=total_iters,
        T_warmup=warmup_iters,
        start_lr=0.02,
        end_lr=0.3,
        mode="linear",
    )
    actual_lr = []
    for _iter in range(total_iters):
        scheduler.step()
        actual_lr.append(optimizer.param_groups[0]["lr"])
    plt.plot(list(range(total_iters)), actual_lr, "--", label="LinearWarmup")

    plt.xlabel("iterations")
    plt.ylabel("learning rate")
    plt.legend()
    plt.savefig("scheduler.png")
