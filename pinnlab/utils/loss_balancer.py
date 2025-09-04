from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LossDict = Dict[str, torch.Tensor]

def _safe_softmax(x: torch.Tensor, dim=-1, T: float = 1.0):
    x = x / max(T, 1e-8)
    x = x - x.max(dim=dim, keepdim=True).values
    return torch.softmax(x, dim=dim)

def _normalize_positive(weights: torch.Tensor, eps: float = 1e-8):
    w = torch.clamp(weights, min=eps)
    return w / (w.sum() + eps)

class BaseBalancer(nn.Module):
    """
    Interface:
      total, weights_dict, info = balancer(losses, step:int, model:nn.Module|None)
    """
    def __init__(self, terms: List[str], log_prefix: str = "w/"):
        super().__init__()
        self.terms = terms
        self.log_prefix = log_prefix

    def forward(self, losses: LossDict, step: int, model: Optional[nn.Module] = None
               ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        raise NotImplementedError

    def extra_params(self) -> List[nn.Parameter]:
        """Params to include in the main optimizer (e.g., for uncertainty weighting)."""
        return []

    def _as_tensor(self, losses: LossDict):
        return torch.stack([losses[k] for k in self.terms], dim=0)

    def _to_dict(self, w: torch.Tensor) -> Dict[str, float]:
        return {f"{self.log_prefix}{k}": float(w[i].detach().cpu()) for i, k in enumerate(self.terms)}

# -------------------------
# 1) Uncertainty Weighting
#   L = sum_i ( 1/(2sigma_i^2) * L_i + log sigma_i )
#   where log_sigma are learnable.
# -------------------------
class UncertaintyWeighting(BaseBalancer):
    def __init__(self, terms: List[str], init_log_sigma: float = 0.0):
        super().__init__(terms)
        # log_sigma are trained jointly with the model
        self.log_sigma = nn.Parameter(torch.full((len(terms),), float(init_log_sigma)))

    def forward(self, losses: LossDict, step: int, model=None):
        L = self._as_tensor(losses)
        # weights in the human-interpretable sense:
        # w_i = 1 / (2 * sigma_i^2) = 0.5 * exp(-2*log_sigma_i)
        w = 0.5 * torch.exp(-2.0 * self.log_sigma).to(L.device)
        # total objective (includes regularizer log sigma)
        total = (w * L).sum() + self.log_sigma.sum()
        w_norm = _normalize_positive(w.detach())
        wd = self._to_dict(w_norm)
        info = {f"sigma/{k}": float(torch.exp(self.log_sigma[i]).detach().cpu())
                for i, k in enumerate(self.terms)}
        return total, wd, info

    def extra_params(self):  # include in main optimizer
        return [self.log_sigma]

# -------------------------
# 2) DWA (Dynamic Weight Averaging)
#   w_i(t) ∝ exp( L_i(t-1)/[L_i(t-2)+eps] / T )
# -------------------------
class DWA(BaseBalancer):
    def __init__(self, terms: List[str], T: float = 2.0):
        super().__init__(terms)
        self.T = T
        self.prev = []  # keep last two loss vectors

    def forward(self, losses: LossDict, step: int, model=None):
        L = self._as_tensor(losses).detach()
        if len(self.prev) < 2:
            # Warmup: equal weights
            w = torch.ones_like(L)
        else:
            r = self.prev[-1] / (self.prev[-2] + 1e-8)
            w = _safe_softmax(r, dim=0, T=self.T)
        total = (w.to(L.device) * self._as_tensor(losses)).sum()
        # Update history
        self.prev.append(L)
        if len(self.prev) > 2: self.prev.pop(0)
        w_norm = _normalize_positive(w)
        return total, self._to_dict(w_norm), {}

# -------------------------
# 3) SoftAdapt
#   w_i ∝ softmax( β * slope_i ), slope_i = L_i(t-1) - L_i(t-2)  (or EWMA slope)
# -------------------------
class SoftAdapt(BaseBalancer):
    def __init__(self, terms: List[str], beta: float = 10.0, use_ewma: bool = True, alpha: float = 0.7):
        super().__init__(terms)
        self.beta = beta
        self.use_ewma = use_ewma
        self.alpha = alpha
        self.hist = None  # running EWMA of losses
        self.prev = None

    def forward(self, losses: LossDict, step: int, model=None):
        L_now = self._as_tensor(losses).detach()
        if self.hist is None:
            self.hist = L_now.clone()
            self.prev = L_now.clone()
            w = torch.ones_like(L_now)
        else:
            if self.use_ewma:
                self.hist = self.alpha * self.hist + (1 - self.alpha) * L_now
                slope = self.hist - self.prev
                self.prev = self.hist.clone()
            else:
                slope = L_now - self.prev
                self.prev = L_now.clone()
            w = _safe_softmax(self.beta * slope, dim=0, T=1.0)
        total = (w.to(L_now.device) * self._as_tensor(losses)).sum()
        w_norm = _normalize_positive(w)
        return total, self._to_dict(w_norm), {}

# -------------------------
# 4) GradNorm-Lite (last layer only, fewer backward passes)
#   Goal: equalize gradient norms per loss on a reference layer.
#   Implementation details:
#     - Mantains positive weights via log-space parameters log_w (no optimizer needed).
#     - Every 'update_every' steps, estimates per-loss grad norms by
#       backwarding each loss on the LAST linear layer only (retain_graph).
#     - Updates weights multiplicatively toward target norms.
# -------------------------
class GradNormLite(BaseBalancer):
    def __init__(self, terms: List[str], alpha: float = 0.12, gamma: float = 0.5,
                 update_every: int = 10, ref_layer_name: Optional[str] = None, clip: float = 5.0):
        super().__init__(terms)
        self.alpha = alpha
        self.gamma = gamma
        self.update_every = update_every
        self.clip = clip
        self.log_w = nn.Parameter(torch.zeros(len(terms)), requires_grad=False)
        self.initial = None
        self.ref_layer_name = ref_layer_name  # optional string to pick layer

    def _pick_last_layer_params(self, model: nn.Module):
        # Prefer explicit name; otherwise take the last nn.Linear params found.
        if self.ref_layer_name:
            for n, p in model.named_parameters():
                if self.ref_layer_name in n and p.requires_grad:
                    return [p]
        last = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            # fallback to last parameter tensor
            params = [p for p in model.parameters() if p.requires_grad]
            return params[-1:]
        return [last.weight] if last.weight.requires_grad else [p for p in last.parameters()]

    def forward(self, losses: LossDict, step: int, model=None):
        assert model is not None, "GradNormLite requires model to measure gradients."
        device = next(model.parameters()).device
        L_vec = self._as_tensor(losses)  # graph-connected tensor
        if self.initial is None:
            self.initial = L_vec.detach()

        w_pos = torch.exp(self.log_w).to(device)
        w_norm = _normalize_positive(w_pos)
        total = (w_norm * L_vec).sum()

        # Periodically update weights to equalize gradient norms
        if step % self.update_every == 0:
            ref_params = self._pick_last_layer_params(model)
            g_norms = []
            # compute grad norm per loss on ref layer
            for i, Li in enumerate(L_vec):
                model.zero_grad(set_to_none=True)
                if self.log_w.requires_grad: self.log_w.grad = None
                (Li).backward(retain_graph=True, inputs=ref_params)
                # L2 norm of grads on ref params
                norms = []
                for p in ref_params:
                    if p.grad is not None:
                        norms.append(p.grad.detach().abs().pow(2).sum())
                g = torch.sqrt(torch.stack(norms).sum() + 1e-12)
                g_norms.append(g)
            G = torch.stack(g_norms)  # shape [num_terms]

            with torch.no_grad():
                # Targets as in GradNorm: G*_i = G_bar * (L_i / L_bar) ** alpha
                G_bar = G.mean()
                L_rel = L_vec.detach() / (self.initial + 1e-8)
                L_bar = L_rel.mean()
                targets = G_bar * (L_rel / (L_bar + 1e-8)).pow(self.alpha)

                ratio = torch.clamp(G / (targets + 1e-8), 1.0 / 10, 10.0)
                # multiplicative update in log-space for stability
                delta = self.gamma * (ratio.log())
                self.log_w -= torch.clamp(delta, -self.clip, self.clip).to(self.log_w.device)
                # Keep weights finite
                self.log_w.clamp_(-6.0, 6.0)

        w_to_report = _normalize_positive(torch.exp(self.log_w.detach()))
        wd = self._to_dict(w_to_report)
        info = {}
        return total, wd, info

# -------------------------
# Factory
# -------------------------
@dataclass
class BalancerConfig:
    kind: str = "none"           # one of: none|uncertainty|dwa|softadapt|gradnorm
    terms: Optional[List[str]] = None
    # kind-specific
    T: float = 2.0               # DWA
    beta: float = 10.0           # SoftAdapt
    ewma_alpha: float = 0.7      # SoftAdapt
    init_log_sigma: float = 0.0  # Uncertainty
    alpha: float = 0.12          # GradNorm-Lite
    gamma: float = 0.5           # GradNorm-Lite
    update_every: int = 10       # GradNorm-Lite
    ref_layer_name: Optional[str] = None
    use_loss_balancer: bool = True

def make_loss_balancer(cfg: BalancerConfig) -> BaseBalancer:
    kind = (cfg.kind or "none").lower()
    if kind == "uncertainty":
        return UncertaintyWeighting(cfg.terms, init_log_sigma=cfg.init_log_sigma)
    if kind == "dwa":
        return DWA(cfg.terms, T=cfg.T)
    if kind == "softadapt":
        return SoftAdapt(cfg.terms, beta=cfg.beta, use_ewma=True, alpha=cfg.ewma_alpha)
    if kind == "gradnorm":
        return GradNormLite(cfg.terms, alpha=cfg.alpha, gamma=cfg.gamma,
                            update_every=cfg.update_every, ref_layer_name=cfg.ref_layer_name)
    # Fallback = no balancing (equal weights)
    class _None(BaseBalancer):
        def forward(self, losses: LossDict, step: int, model=None):
            L = self._as_tensor(losses)
            w = torch.ones_like(L)
            total = (w * self._as_tensor(losses)).sum()
            return total, self._to_dict(_normalize_positive(w)), {}
    return _None(cfg.terms or [])