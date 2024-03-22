import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import random
import math
from torch.cuda import nvtx
import torchsummary
from thop import profile
from einops import einsum

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        self.feats = feats
        self.head = head
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats//head)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats//head)
        self.mlp = nn.Sequential(
            GroupedLinear(feats, mlp_hidden, num_groups = head),
            nn.GELU(),
            nn.Dropout(dropout),
            GroupedLinear(mlp_hidden, feats, num_groups = head),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        nvtx.range_push('model forward_split')
        breakpoint()
        b, n, f = x.size()
        x = x.view(b, n, self.head, self.feats//self.head)  # (b,n,h,f)
        out = self.msa(self.la1(x)) + x                     # (b,n,h,f)
        out = self.mlp(self.la2(out)) + out                 # (b,n,h,f) -> (b,n,h,4f) ->(b,n,h,f)
        nvtx.range_pop()
        return out.flatten(2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.shuffle_order = list(range(feats)) #random shuffle

        # # Inverse shuffle order
        # self.inverse_shuffle_order = [0] * feats #random shuffle
        # for i, j in enumerate(self.shuffle_order): #random shuffle
        #     self.inverse_shuffle_order[j] = i #random shuffle
                
        # random.shuffle(self.shuffle_order) #random shuffle
        self.q = GroupedLinear(feats, feats, num_groups= head)
        self.k = GroupedLinear(feats, feats, num_groups= head)
        self.v = GroupedLinear(feats, feats, num_groups= head)

        self.o = GroupedLinear(feats, feats, num_groups= head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #batch, seq_len, dim///
        b, n, h, f = x.size()
        # x = x[:, :, self.shuffle_order] #Random shuffling
        breakpoint()
        x = x.transpose(2,3).reshape(b,n,h,f)
        q = self.q(x).transpose(1,2)
        k = self.k(x).transpose(1,2)
        v = self.v(x).transpose(1,2)

        # nvtx.range_push('Attention + score')
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        # nvtx.range_pop()
        o = self.dropout(self.o(attn))
        return o

from torch import Tensor


class GroupedLinear(nn.Module):
    __constants__ = ['in_features', 'out_features', 'num_groups']
    in_features: int
    out_features: int
    num_groups: int
    weight: Tensor
    def __init__(self, in_features: int, out_features: int, num_groups: int, device=None, dtype=None, bias: bool = True,) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GroupedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        assert in_features % num_groups == 0, "in_features must be divisible by groups"
        assert out_features % num_groups == 0, "out_features must be divisible by groups"
        self.weight = nn.Parameter(torch.empty((num_groups, in_features // num_groups, out_features // num_groups), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features//num_groups, **factory_kwargs))
        else:
            self.register_parameter('bias', None)        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for g in range(self.num_groups):
            nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            for g in range(self.num_groups):
                nn.init.uniform_(self.bias[g], -bound, bound)

    def forward(self, x):
        # breakpoint()
        # x = (.., h, f//h)
        # Apply each linear layer to its corresponding group
        breakpoint()
        out = torch.einsum("...gi, gij->...gj", x, self.weight)
        if self.bias is not None:
            out += self.bias
        return out


if __name__=="__main__":
    b,n,h,f = 1, 36, 3, 330
    x = torch.randn(b,n,f).cpu()
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f, f*4, h).cpu()
    torchsummary.summary(net, (n,f), device= 'cpu')
    out = net(x)
    print(out.shape)
    import numpy as np

    def calculate_y(x):
        return 1.73205080756888 * np.sqrt(x**2 + x + 0.75) - 1.5
    print(calculate_y(192))

    flops, params = profile(net, inputs=(x, ))
    print(f'flops: {flops}, params: {params}')
    # b,n,h,f = 4, 6, 6,60
    # grouped_linear = GroupedLinear(f, f, h, bias=False)

    # # Create a test input tensor
    # x = torch.randn(b, n, h, f // h)
    # new_weights = torch.randn(h,f // h, f // h)
    # # Set all inputs to 0 except for one group to test
    # test_group = 1
    # for g in range(h):
    #     if g != test_group:
    #         x[:,:, g, :] = 0        
    #         new_weights[g,:,:] = 0    
    #     else:
    #         new_weights[g,:,:] = 1
    # grouped_linear.weight = nn.Parameter(new_weights)
        
    # for g in range(h):
    #     # print(x[:,g,:])
    #     print(grouped_linear.weight[g,:,:])
    # # Perform the forward pass
    # output = grouped_linear(x)

    # # Print the test output
    # for g in range(h):
    #     print(output[:,:,g,:])