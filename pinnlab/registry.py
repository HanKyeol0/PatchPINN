# models
from pinnlab.models.mlp import MLP
from pinnlab.models.fourier_mlp import FourierMLP
from pinnlab.models.residual_network import ResidualNetwork
from pinnlab.models.patch_transformer import PatchTransformer
from pinnlab.models.patch_attention import PatchAttention

# experiments
from pinnlab.experiments.burgers1d import Burgers1D
from pinnlab.experiments.helmholtz2d import Helmholtz2D
from pinnlab.experiments.helmholtz2d_steady import Helmholtz2DSteady
from pinnlab.experiments.poisson2d import Poisson2D
from pinnlab.experiments.navierstokes2d import NavierStokes2D
from pinnlab.experiments.convection1d import Convection1D
from pinnlab.experiments.reactiondiffusion1d import ReactionDiffusion1D
from pinnlab.experiments.reactiondiffusion2d import ReactionDiffusion2D
from pinnlab.experiments.allencahn1d import AllenCahn1D
from pinnlab.experiments.allencahn2d import AllenCahn2D

# patch-based experiment
from pinnlab.experiments.poisson2d_patch import Poisson2D_Patch

_MODEL_REG = {
    "mlp": MLP,
    "fourier_mlp": FourierMLP,
    "residual_network": ResidualNetwork,
    "patch_transformer": PatchTransformer,
    "patch_attention": PatchAttention,
}

_EXP_REG = {
    "burgers1d": Burgers1D,
    "helmholtz2d": Helmholtz2D,
    "helmholtz2d_steady": Helmholtz2DSteady,
    "poisson2d": Poisson2D,
    "navierstokes2d": NavierStokes2D,
    "convection1d": Convection1D,
    "reactiondiffusion1d": ReactionDiffusion1D,
    "reactiondiffusion2d": ReactionDiffusion2D,
    "allencahn1d": AllenCahn1D,
    "allencahn2d": AllenCahn2D,
    "poisson2d_patch": Poisson2D_Patch,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]
