import torch

class Interval:
    def __init__(self, x_min, x_max, device): self.a, self.b, self.device = x_min, x_max, device
    def sample(self, n): 
        return torch.rand(n, 1, device=self.device) * (self.b - self.a) + self.a

class Rectangle:
    def __init__(self, x_min, x_max, y_min, y_max, device):
        self.xa, self.xb, self.ya, self.yb, self.device = x_min, x_max, y_min, y_max, device
    def sample(self, n):
        x = torch.rand(n,1,device=self.device)*(self.xb-self.xa)+self.xa
        y = torch.rand(n,1,device=self.device)*(self.yb-self.ya)+self.ya
        return torch.cat([x,y], dim=1)

def linspace_2d(xa, xb, ya, yb, nx, ny, device):
    x = torch.linspace(xa, xb, nx, device=device)
    y = torch.linspace(ya, yb, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    return X, Y
