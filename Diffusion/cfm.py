import math
import torch

from optimal_transport import OTPlanSampler

class ConditionalFlowMatcher:
    def __init__(self, sigma: float = 0.0):
        self.sigma = torch.tensor(sigma)

    def compute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path: t * x1 + (1 - t) * x0"""
        t = t.view(-1, *([1] * (len(x0.shape) - 1)))
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """Return sigma for constant noise"""
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """Sample from the probability path N(mu_t, sigma)"""
        mu_t = self.compute_mu_t(x0, x1, t).
        sigma_t = self.compute_sigma_t(t).view(-1, *([1] * (len(x0.shape) - 1)))
        return mu_t.to(x0.device) + sigma_t.to(x0.device) * epsilon.to(x0.device)

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field ut(x1|x0) = x1 - x0"""
        return x1 - x0

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        epsilon = torch.randn_like(x0)
        xt = self.sample_xt(x0, x1, t, epsilon)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        return t, xt, ut

class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        """Compute the mean using trigonometric interpolation"""
        t = t.view(-1, *([1] * (len(x0.shape) - 1)))
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        """Conditional vector field using trigonometric interpolation"""
        t = t.view(-1, *([1] * (len(x0.shape) - 1)))
        return (math.pi / 2) * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


class VarianceExplodingConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma=0.0, lmbd=1):
        super().__init__(sigma)
        self.lmbd = lmbd

    def compute_mu_t(self, x0, x1, t):
        return x0

    def compute_sigma_t(self, t):
        return self.sigma * torch.sqrt(torch.exp(self.lmbd * t) - 1)


class OptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        """Draw samples based on OT plan coupling x0, x1"""
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t)
