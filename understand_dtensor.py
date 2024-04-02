import torch

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.d_tensor import ShardingSpec
from colossalai.tensor.d_tensor import distribute_tensor

colossalai.launch_from_torch(config={})

# define your device mesh
# assume you have 4 GPUs
physical_mesh_id = torch.arange(0, 4)
mesh_shape = (2, 2)
device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=False)

# define a tensor
a = torch.rand(16, 32).cuda()

# create sharding spec for the tensor
# assume the sharding spec is [S0, R]
dim_partition_dict = {0: [0]}
sharding_spec = ShardingSpec(a.dim(), dim_partition_dict)

# create a distributed tensor
# d_tensor = DTensor(a, device_mesh, sharding_spec)
d_tensor = distribute_tensor(a, device_mesh, sharding_spec)
print(d_tensor)

global_tensor = d_tensor.to_global()
print(global_tensor)