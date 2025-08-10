# models
from pinnlab.models.mlp import MLP
from pinnlab.models.fourier_mlp import FourierMLP

# experiments
from pinnlab.experiments.burgers1d import Burgers1D
from pinnlab.experiments.helmholtz2d import Helmholtz2D
from pinnlab.experiments.poisson2d import Poisson2D

_MODEL_REG = {
    "mlp": MLP,
    "fourier_mlp": FourierMLP,
}

_EXP_REG = {
    "burgers1d": Burgers1D,
    "helmholtz2d": Helmholtz2D,
    "poisson2d": Poisson2D,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]
