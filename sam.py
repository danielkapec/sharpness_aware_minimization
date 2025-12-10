import torch
from torch.optim import Optimizer

class SAM(Optimizer):

    def __init__(self, params, base_optimizer, rho = 0.05, adaptive: bool = False, **kwargs):
        assert rho >= 0.0, "Invalid rho, should be nonnegative"
        self.rho = rho
        self.adaptive = adaptive

        defaults = dict(rho = rho, adaptive = adaptive, **kwargs)
        super().__init__(params, defaults)

        if isinstance(base_optimizer, type):
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        else:
            raise ValueError("base_optimizer must be an optimizer class")
        
        self.param_groups = self.base_optimizer.param_groups


    @torch.no_grad()
    def _grad_norm(self):
        norms = []
        device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if self.adaptive:
                    g = g * p.abs()
                norms.append(g.norm(p = 2))
        if not norms:
            return torch.tensor(0.0, device = device)
        return torch.norm(torch.stack(norms), p = 2)




    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if self.adaptive:
                    e_w = p.grad * p.abs() * scale
                else:
                    e_w = p.grad * scale

                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        raise RuntimeError("sam does not support step")
