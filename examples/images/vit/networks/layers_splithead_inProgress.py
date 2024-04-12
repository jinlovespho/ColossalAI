import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import random
import math
from torch.cuda import nvtx
import torchsummary


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

class FeedForward(nn.Module):
    def __init__(self, ffn_feats, mlp_hidden, n_head, ffn_dropout):
        super(FeedForward,self).__init__()
        
        self.n_head = n_head
        self.ffn_feats = ffn_feats 
        # self.mlp_hidden = mlp_hidden 
        self.ffn_h_dim = int(self.ffn_feats // self.n_head)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.gelu=nn.GELU()
        
        self.ffn_linear1 = nn.ModuleList([nn.Linear(self.ffn_h_dim, self.ffn_h_dim*4) for _ in range(self.n_head)])
        self.ffn_linear2 = nn.ModuleList([nn.Linear(self.ffn_h_dim*4, self.ffn_h_dim) for _ in range(self.n_head)])

    def forward(self, x):
        b, n, h, f = x.shape
        
        # breakpoint()
        tmp1=torch.zeros(b,n,h,f*4).to('cuda')
        for i in range(self.n_head):
            tmp1[:,:,i:i+1,:] = self.ffn_linear1[i](x[:,:,i:i+1,:])
        tmp1 = self.ffn_dropout(self.gelu(tmp1))
        
        tmp2 = torch.zeros(b,n,h,f).to('cuda')
        for i in range(self.n_head):
            tmp2[:,:,i:i+1,:] = self.ffn_linear2[i](tmp1[:,:,i:i+1,:])
        tmp2 = self.ffn_dropout(self.gelu(tmp2))

        return tmp2


# Method = 0
class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., splithead_method:int=0):
        # breakpoint()
        self.feats = feats
        self.head = head
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats//head)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats//head)  
        self.mlp = FeedForward(feats, mlp_hidden, head, dropout)

        self.Method = splithead_method
        if self.Method==0:
            print("Method: Linear")
            self.Merge = nn.Linear(feats, feats)
        elif self.Method==1:
            print("Method: Feature-wise Linear")
            self.Merge = nn.Linear(feats//head,feats//head)
        elif self.Method==2:
            print("Method: Feature-wise Conv")
            self.Merge = nn.Linear(head,head)
        elif self.Method==3:
            print("Method: Shuffle")
        elif self.Method==4:
            self.Shift=2
            print(f"Method: Shift-(1/{self.Shift})")

    def forward(self, x: torch.Tensor):
        nvtx.range_push('model forward_split')
        # breakpoint()
        b, n, f = x.size()
        x = x.view(b, n, self.head, self.feats//self.head)
        out = self.msa(self.la1(x)) + x 
        out = self.mlp(self.la2(out)) + out
        
        # breakpoint()
        
        if self.Method==0:   
            # Linear
            out = out.flatten(2)
            out = self.Merge(out)
            # out = out.reshape(b,n, self.head, self.feats//self.head)            
        elif self.Method==1:
            # Feature-wise Linear
            out = self.Merge(out)
        elif self.Method==2:
            # Feature-wise Conv
            out = out.transpose(2,3) # b,n,f,h
            out = self.Merge(out)
            out = out.transpose(2,3) # b,n,h,f
        elif self.Method==3:
            # Feature Shuffle
            out = out.transpose(2,3).reshape(b,n,self.head, self.feats//self.head) # shuffle
        elif self.Method==4:
            # Feature Shift
            out = torch.roll(out, shifts=self.feats//self.Shift, dims=-1)
        nvtx.range_pop()
        return out.flatten(2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.head_dim = int(self.feats // self.head)
        
        self.q_head_linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(head)])
        self.k_head_linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(head)])
        self.v_head_linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(head)])
        self.attn_out_linears = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(head)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #batch, seq_len, dim///
        b, n, h, f = x.size()

        q=torch.zeros(b, n, self.head, self.head_dim).to('cuda')
        k=torch.zeros(b, n, self.head, self.head_dim).to('cuda')
        v=torch.zeros(b, n, self.head, self.head_dim).to('cuda')
        
        for i in range(self.head):
            q[:,:,i:i+1,:] = self.q_head_linears[i](x[:,:,i:i+1,:])
            k[:,:,i:i+1,:] = self.k_head_linears[i](x[:,:,i:i+1,:])
            v[:,:,i:i+1,:] = self.v_head_linears[i](x[:,:,i:i+1,:])

        ForkedPdb().set_trace()
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        # nvtx.range_push('Attention + score')
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        # nvtx.range_pop()
        
        for i in range(self.head):
            attn[:,:,i:i+1,:] = self.attn_out_linears[i](attn[:,:,i:i+1,:])
        
        return attn

from torch import Tensor


# class GroupedLinear(nn.Module):
#     __constants__ = ['in_features', 'out_features', 'num_groups']
#     in_features: int
#     out_features: int
#     num_groups: int
#     weight: Tensor
#     def __init__(self, in_features: int, out_features: int, num_groups: int, device=None, dtype=None, bias: bool = True,) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(GroupedLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_groups = num_groups
#         assert in_features % num_groups == 0, "in_features must be divisible by groups"
#         assert out_features % num_groups == 0, "out_features must be divisible by groups"
#         self.weight = nn.Parameter(torch.empty((num_groups, in_features // num_groups, out_features // num_groups), **factory_kwargs))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(num_groups, out_features//num_groups, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)        
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         for g in range(self.num_groups):
#             nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             for g in range(self.num_groups):
#                 nn.init.uniform_(self.bias[g], -bound, bound)

#     def forward(self, x):
#         # x = (.., g, f//g)
#         # Apply each linear layer to its corresponding group
#         # breakpoint()
#         out = torch.einsum("...gi, gij->...gj", x, self.weight)
#         if self.bias is not None:
#             out += self.bias
#         return out


if __name__=="__main__":
    b,n,h,f = 1, 36, 3, 192
    x = torch.randn(b,n,f).cpu()
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f, f*4, h).cpu()
    torchsummary.summary(net, (n,f), device= 'cpu')
    out = net(x)
    print(out.shape)

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