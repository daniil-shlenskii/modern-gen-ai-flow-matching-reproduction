from typing import Callable

import torch
from torch.distributions.normal import Normal

from utils import divergence


def compute_nll(
    x: torch.Tensor,
    drift_fn: Callable,
    start_time: float = 1.0,
    end_time: float = 0.0,
    n_ode_steps: int = 1,
) -> torch.Tensor:
    x = x.clone()
    ode_steps = torch.linspace(start_time, end_time, steps=n_ode_steps+1)

    div_sum_h = 0
    for i, t in enumerate(ode_steps[:-1]):
        x.requires_grad_(True)

        t_next = ode_steps[i + 1]
        h = t_next - t

        drifts = drift_fn(x, t.expand_as(x))
        div_sum_h += divergence(drifts, x) * h
        x = x + drifts * h
    
    l0 = Normal(
        torch.zeros_like(x), torch.ones_like(x),
    ).log_prob(x)
    l0 = l0.flatten(1).sum(-1)
    return -(l0 + div_sum_h)
