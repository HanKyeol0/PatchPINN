# models
from pinnlab.models.attention import Attention
from pinnlab.models.ffn import FFN
from pinnlab.models.ffn_context import FFNContext
from pinnlab.models.cnn import CNN, SimpleCNN
from pinnlab.models.unet import UNetCNN
from pinnlab.models.residual_ffn import ResidualFFN

# experiments
from pinnlab.experiments.helmholtz2d_steady import Helmholtz2DSteady
from pinnlab.experiments.helmholtz2d import Helmholtz2D
from pinnlab.experiments.allencahn2d import AllenCahn2D
from pinnlab.experiments.poisson2d import Poisson2D
from pinnlab.experiments.navierstokes2d import NavierStokes2D
from pinnlab.experiments.burgers2d import Burgers2D

_MODEL_REG = {
    "ffn": FFN,
    "residual_ffn": ResidualFFN,
    "attention": Attention,
    "ffn_context": FFNContext,
    "cnn": CNN,
    "unet": UNetCNN,
    "simple_cnn": SimpleCNN,
}

_EXP_REG = {
    "helmholtz2d_steady": Helmholtz2DSteady,
    "helmholtz2d": Helmholtz2D,
    "allencahn2d": AllenCahn2D,
    "poisson2d": Poisson2D,
    "navierstokes2d": NavierStokes2D,
    "burgers2d": Burgers2D,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]