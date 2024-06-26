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
            

__all__ = ["ViTPolicy", "ViTModelPolicy", "ViTForImageClassificationPolicy", "ViTForMaskedImageModelingPolicy"]


class ViTPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTOutput, ViTSelfAttention
        
        # ForkedPdb().set_trace()
        policy = {}

        # 실행 X
        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Vit doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        # 실행 O
        if self.shard_config.enable_tensor_parallelism:
            policy[ViTEmbeddings] = ModulePolicyDescription(
                attribute_replacement={},
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForReplicatedInput,
                    )
                ],
            )
            # ForkedPdb().set_trace()

            policy[ViTLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.attention.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",     # Linear(hidden,hidden)
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",       # Linear(hidden,hidden)
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",     # Linear(hidden,hidden)
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(            # Linear(hidden,hidden)
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                    SubModuleReplacementDescription(            
                        suffix="intermediate.dense",                # Linear(hidden, 4*hidden)
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(                # Linear(4*hidden, hidden)
                        suffix="output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(                # Dropout()
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
            )
            
            # ForkedPdb().set_trace()

        # use flash attention
        if self.shard_config.enable_flash_attention:    # 실행 X
            self.append_or_create_method_replacement(
                description={
                    "forward": get_vit_flash_self_attention_forward(),
                },
                policy=policy,
                target_key=ViTSelfAttention,
            )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:          # 실행 X
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_vit_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=ViTOutput,
            )
        # ForkedPdb().set_trace()
        return policy

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        if self.model.__class__.__name__ == "ViTModel":
            module = self.model
        else:
            module = self.model.vit
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, pipeline_forward: Callable, policy: Dict):
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "ViTModel":
                module = self.model
            else:
                module = self.model.vit

            layers_per_stage = Policy.distribute_layers(len(module.encoder.layer), stage_manager.num_stages)
            stage_index = Policy.get_stage_index(layers_per_stage, stage_manager.stage)
            method_replacement = {"forward": pipeline_forward(stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )


# ViTModel
class ViTModelPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTModel

        policy = super().module_policy()
        
        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(module.pooler)

        return held_layers


# ViTForImageClassification
class ViTForImageClassificationPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTModel

        policy = super().module_policy()        # 부모의 policy를 먼저 가져오고, 아래에 policy.update()으로 new policy를 추가해준다.
        # ForkedPdb().set_trace()
        if self.shard_config.enable_tensor_parallelism:
            new_item = {
                ViTForImageClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(        # classifier -> Linear(384,100)
                            suffix="classifier", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            # ForkedPdb().set_trace()
            policy.update(new_item)     # update는 내장 파이썬 dictionary 함수
            # ForkedPdb().set_trace()

        # 실행 X
        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
            self.set_pipeline_forward(
                model_cls=ViTForImageClassification,
                pipeline_forward=ViTForImageClassification_pipeline_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model.vit
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(self.model.classifier)

        return held_layers


# ViTForMaskedImageModeling
class ViTForMaskedImageModelingPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForMaskedImageModeling, ViTModel

        policy = super().module_policy()

        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
            self.set_pipeline_forward(
                model_cls=ViTForMaskedImageModeling,
                pipeline_forward=ViTForMaskedImageModeling_pipeline_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model.vit
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(self.model.decoder)

        return held_layers
