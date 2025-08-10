import torch, numpy as np, math
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Interval

class Convection1D_1(BaseExperiment):
    """
    1D Convection Equation

    ∂u/∂t + β ∂u/∂x = 0, ∀x ∈ [0, 2π], t ∈ [0, 1]
    IC:u(x, 0) = sin(x), BC:u(0, t) = u(2π, t)
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.time_dep = bool(cfg.get("time_dependent", False))
        a, b = cfg["domain"]["x"] # range
        self.x_dom = Interval(a, b, device)
        self.t0, self.t1 = cfg["domain"]["t"]
        self.beta = eval(str(cfg["beta"])) if isinstance(cfg["beta"], str) else float(cfg["beta"])

    def sample_batch(self, n_f, n_b, n_0):
        X_f = torch.cat([
            self.x_dom.sample(n_f),
            torch.rand(n_f,1,device=self.device)*(self.t1-self.t0)+self.t0
        ], dim=1)

        t_b = torch.rand(n_b,1,device=self.device)*(self.t1-self.t0)+self.t0
        xa = torch.full((n_b,1), self.x_dom.a, device=self.device)
        xb = torch.full((n_b,1), self.x_dom.b, device=self.device)
        X_b = torch.cat([torch.cat([xa,t_b],1), torch.cat([xb,t_b],1)], dim=0)
        u_b = torch.zeros(X_b.size(0),1, device=self.device)

        x0 = self.x_dom.sample(n_0)
        t0 = torch.full((n_0,1), self.t0, device=self.device)
        X_0 = torch.cat([x0,t0], dim=1)
        u0 = -torch.sin(math.pi*x0)

        return {"X_f": X_f, "X_b": X_b, "u_b": u_b, "X_0": X_0, "u0": u0}
    
    def pde_residual_loss(self, model, batch):
        X = make_leaf(batch["X_f"])             # Leaf w/ grad
        u = model(X)                            # [N,1]
        du = grad_sum(u, X)                     # [N,2] => [du/dx, du/dt]
        u_x, u_t = du[:,0:1], du[:,1:2]
        res = u_t + self.beta * u_x
        return res.pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        X0, u0 = batch["X_0"], batch["u0"]
        pred = model(X0)
        return (pred - u0).pow(2)