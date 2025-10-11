# models
from pinnlab.models.patch_attention import PatchAttention
from pinnlab.models.patch_ffn import PatchFFN
from pinnlab.models.patch_ffn_context import PatchFFNContext
from pinnlab.models.patch_cnn import PatchCNN, SimplePatchCNN
from pinnlab.models.patch_unet import UNetPatchCNN
from pinnlab.models.patch_ffn_hard_bc import PatchFFNHardBC

# experiments
from pinnlab.experiments.helmholtz2d_steady_patch import Helmholtz2DSteady_patch
from pinnlab.experiments.helmholtz2d_patch import Helmholtz2D_patch

_MODEL_REG = {
    "patch_attention": PatchAttention,
    "patch_ffn": PatchFFN,
    "patch_ffn_context": PatchFFNContext,
    "patch_cnn": PatchCNN,
    "patch_unet": UNetPatchCNN,
    "patch_ffn_hard_bc": PatchFFNHardBC,
    "simple_patch_cnn": SimplePatchCNN,
}

_EXP_REG = {
    # "burgers1d": Burgers1D,
    # "helmholtz2d": Helmholtz2D,
    # "helmholtz2d_steady": Helmholtz2DSteady,
    # "poisson2d": Poisson2D,
    # "navierstokes2d": NavierStokes2D,
    # "convection1d": Convection1D,
    # "reactiondiffusion1d": ReactionDiffusion1D,
    # "reactiondiffusion2d": ReactionDiffusion2D,
    # "allencahn1d": AllenCahn1D,
    # "allencahn2d": AllenCahn2D,
    "helmholtz2d_steady_patch": Helmholtz2DSteady_patch,
    "helmholtz2d_patch": Helmholtz2D_patch,
}

def get_model(name):     return _MODEL_REG[name]
def get_experiment(name):return _EXP_REG[name]