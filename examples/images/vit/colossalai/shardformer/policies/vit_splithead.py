import warnings
from typing import Callable, Dict, List, Union

import torch.nn as nn

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.layer import DropoutForReplicatedInput, Linear1D_Col

from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.vit import (
    ViTForImageClassification_pipeline_forward,
    ViTForMaskedImageModeling_pipeline_forward,
    ViTModel_pipeline_forward,
    get_jit_fused_vit_output_forward,
    get_vit_flash_self_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            

class ViT_SplitHead_Policy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}
    
        return policy

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        pass

    def set_pipeline_forward(self, model_cls: nn.Module, pipeline_forward: Callable, policy: Dict):
        pass
