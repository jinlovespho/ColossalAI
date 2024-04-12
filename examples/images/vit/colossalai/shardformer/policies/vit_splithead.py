import warnings
from typing import Callable, Dict, List, Union

import torch.nn as nn

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.layer import DropoutForReplicatedInput, DropoutForParallelInput, Linear1D_Col, Linear1D_Row

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
        from networks.vit_splithead import ViT_SplitHead
        from networks.layers_splithead import TransformerEncoder
        
        # ForkedPdb().set_trace()
        policy = {}

        # 실행 X
        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Vit doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        # ForkedPdb().set_trace()
        # 실행 O
        if self.shard_config.enable_tensor_parallelism:
            # self.model <class 'networks.vit_splithead.ViT_SplitHead'>
            # ForkedPdb().set_trace()
            
            n_head = self.model.enc[0].msa.head
            hidden_dim = self.model.enc[0].msa.feats
            head_dim = self.model.enc[0].msa.head_dim
            
            trans_enc_sub_mod_replace=[]
            for i in range(n_head):
                trans_enc_sub_mod_replace.append( SubModuleReplacementDescription(suffix=f'msa.q_head_linears.{i}', target_module=Linear1D_Col))
                trans_enc_sub_mod_replace.append( SubModuleReplacementDescription(suffix=f'msa.k_head_linears.{i}', target_module=Linear1D_Col))
                trans_enc_sub_mod_replace.append( SubModuleReplacementDescription(suffix=f'msa.v_head_linears.{i}', target_module=Linear1D_Col))
                trans_enc_sub_mod_replace.append( SubModuleReplacementDescription(suffix=f'msa.attn_out_linears.{i}', target_module=Linear1D_Row))
            
            trans_enc_sub_mod_replace.append( SubModuleReplacementDescription(suffix='msa.dropout', target_module=DropoutForParallelInput))
            
            # ForkedPdb().set_trace()
            
            policy[TransformerEncoder] = ModulePolicyDescription(
                attribute_replacement={
                    "msa.head": n_head // self.shard_config.tensor_parallel_size,
                    "msa.feats": hidden_dim // self.shard_config.tensor_parallel_size,
                    'msa.sqrt_d': (hidden_dim//self.shard_config.tensor_parallel_size)**0.5,
                    "msa.head_dim": head_dim // self.shard_config.tensor_parallel_size
                },
                param_replacement=[],
                sub_module_replacement=trans_enc_sub_mod_replace, 
            )
                    
            #         # FFN
            #         SubModuleReplacementDescription(
            #             suffix="mlp.0",                 # Linear(hidden, 4*hidden), mlp[0]해야할 수도 
            #             target_module=Linear1D_Col,
            #         ),
            #         SubModuleReplacementDescription(
            #             suffix="mlp.3",                 # Linear(4*hidden, hidden), mlp[3]해야할 수도
            #             target_module=Linear1D_Row,
            #         ),
            #         SubModuleReplacementDescription(               
            #             suffix="mlp.5",                  # Dropout()
            #             target_module=col_nn.DropoutForReplicatedInput,
            #         ),
            #     ]
            # )
            
            # policy[ViT] = ModulePolicyDescription(
            #     attribute_replacement={},
            #     param_replacement=[],
            #     sub_module_replacement=[
            #         SubModuleReplacementDescription(
            #             suffix="fc.1",         # Linear(hidden, num_class)
            #             target_module=Linear1D_Col,
            #             kwargs=dict(gather_output=True)
            #         )
            #     ]
            # )
            
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
