import os
from typing import Dict, List, Tuple

import torch.nn as nn
from torch import Tensor

from colossalai.cluster import DistCoordinator

from ..policies.base_policy import Policy
from .shard_config import ShardConfig
from .sharder import ModelSharder

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
            

# set CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that when communication and computation overlap, the order of core scheduling is correct
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


class ShardFormer:
    """
    Parallelize model based on the given config and policy

    Example:

    ```python
    from colossalai.shardformer import ShardFormer, ShardConfig
    from transformers import BertForMaskedLM
    import colossalai
    import torch

    colossalai.launch_from_torch(config={})

    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    shard_config = ShardConfig()
    shard_former = ShardFormer(shard_config=shard_config)
    model, shared_params = shard_former.optimize(org_model)
    ```
    """

    def __init__(self, shard_config: ShardConfig):
        # breakpoint()
        self.coordinator = DistCoordinator()
        self.shard_config = shard_config

    def optimize(self, model: nn.Module, policy: Policy = None) -> Tuple[nn.Module, List[Dict[int, Tensor]]]:
        r"""
        This method will optimize the model based on the given policy.

        Args:
            model (`torch.nn.Model`): the origin huggingface model
            shard_config (`ShardConfig`): the config for distribute information
            policy (`Policy`): the custom policy for sharding

        Returns: the sharded model and the shared parameters
        """
        # ForkedPdb().set_trace()
        sharder = ModelSharder(model=model, shard_config=self.shard_config, policy=policy)
        # ForkedPdb().set_trace()
        shared_params = sharder.shard()
        # ForkedPdb().set_trace()
        return model, shared_params
