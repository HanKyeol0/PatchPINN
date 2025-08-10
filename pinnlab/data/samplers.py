import torch
def concat_time(X, t):
    if t is None: return X
    if X.dim()==1: X = X[:,None]
    if t.dim()==1: t = t[:,None]
    if X.shape[0] != t.shape[0]:
        t = t.expand(X.shape[0], -1)
    return torch.cat([X, t], dim=1)

def uniform_time(n, t0, t1, device):
    return torch.rand(n,1,device=device)*(t1-t0)+t0
