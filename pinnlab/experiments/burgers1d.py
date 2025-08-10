import torch, numpy as np, math
from pinnlab.experiments.base import BaseExperiment, make_leaf, grad_sum
from pinnlab.data.geometries import Interval

class Burgers1D(BaseExperiment):
    """
    u_t + u u_x - nu u_xx = 0,  x in [a,b], t in [0,1]
    IC: u(x,0) = -sin(pi x)
    BC: u(a,t)=u(b,t)=0
    """
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        a, b = cfg["domain"]["x"]
        self.x_dom = Interval(a, b, device)
        self.t0, self.t1 = cfg["domain"]["t"]
        self.nu = eval(str(cfg["nu"])) if isinstance(cfg["nu"], str) else float(cfg["nu"])

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
        d2u_dx = grad_sum(u_x, X)               # [N,2]
        u_xx = d2u_dx[:,0:1]
        res = u_t + u * u_x - self.nu * u_xx
        return res.pow(2)

    def boundary_loss(self, model, batch):
        Xb, ub = batch["X_b"], batch["u_b"]
        pred = model(Xb)
        return (pred - ub).pow(2)

    def initial_loss(self, model, batch):
        X0, u0 = batch["X_0"], batch["u0"]
        pred = model(X0)
        return (pred - u0).pow(2)

    def true_solution(self, x, t):
        return -torch.sin(math.pi*x)  # no exact solution yet...

    def relative_l2_on_grid(self, model, grid_cfg):
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        with torch.no_grad():
            U = model(XT).reshape(nx, nt)
        U0_true = self.true_solution(xs[:,None], torch.zeros_like(xs[:,None]))[:,0]
        U0_pred = U[:,0]
        rel = torch.linalg.norm(U0_pred - U0_true)/torch.linalg.norm(U0_true)
        return rel.item()

    def plot_final(self, model, grid_cfg, out_dir):
        nx, nt = grid_cfg["nx"], grid_cfg["nt"]
        xs = torch.linspace(self.x_dom.a, self.x_dom.b, nx, device=self.device)
        ts = torch.linspace(self.t0, self.t1, nt, device=self.device)
        X, T = torch.meshgrid(xs, ts, indexing="ij")
        XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)
        with torch.no_grad():
            U_pred = model(XT).reshape(nx, nt).cpu().numpy()
        import numpy as np
        U_true = np.zeros_like(U_pred)
        U_true[:,0] = (-torch.sin(math.pi*xs)).cpu().numpy()
        from pinnlab.utils.plotting import save_plots_1d
        return save_plots_1d(xs.cpu().numpy(), ts.cpu().numpy(), U_true, U_pred, out_dir, "burgers1d")
