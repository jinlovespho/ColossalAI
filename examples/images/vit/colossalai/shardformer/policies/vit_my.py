import warnings
from typing import Callable, Dict, List, Union

import torch.nn as nn

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.layer import DropoutForReplicatedInput, Linear1D_Col, Linear1D_Row

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

__all__ = ["ViT_My_Policy"]
            


class ViT_My_Policy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from networks.vit import ViT
        from networks.layers import TransformerEncoder
        
        # ForkedPdb().set_trace()
        policy = {}

        # 실행 X
        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Vit doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        # 실행 O
        if self.shard_config.enable_tensor_parallelism:
            # self.model : <'networks.vit.ViT'>
            # ForkedPdb().set_trace()
            
            policy[TransformerEncoder] = ModulePolicyDescription(
                attribute_replacement={
                    "msa.head": self.model.enc[0].msa.head // self.shard_config.tensor_parallel_size,
                    "msa.tp_feats": self.model.enc[0].msa.feats // self.shard_config.tensor_parallel_size,
                },
                param_replacement=[],
                sub_module_replacement=[
                    # Self-ATTN
                    SubModuleReplacementDescription(
                        suffix="msa.q",         # Linear(hidden,hidden)
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="msa.k",         # Linear(hidden,hidden)
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="msa.v",         # Linear(hidden,hidden)
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="msa.dropout",         # Dropout()
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="msa.o",                # Linear(hidden,hidden)
                        target_module=Linear1D_Row,
                    ),
                    
                    # FFN
                    SubModuleReplacementDescription(
                        suffix="mlp.0",                 # Linear(hidden, 4*hidden), mlp[0]해야할 수도 
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.3",                 # Linear(4*hidden, hidden), mlp[3]해야할 수도
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(               
                        suffix="mlp.5",                  # Dropout()
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ]
            )
            
            policy[ViT] = ModulePolicyDescription(
                attribute_replacement={},
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="fc.1",         # Linear(hidden, num_class)
                        target_module=Linear1D_Col,
                        kwargs=dict(gather_output=True)
                    )
                ]
            )
            
            # ForkedPdb().set_trace()
            
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
