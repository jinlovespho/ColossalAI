import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import random
import math
from torch.cuda import nvtx
import torchsummary
from thop import profile
from einops import rearrange




# ONLY CLS TKNS REPLACED 
class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        self.feats = feats
        self.head = head
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats//head)
        self.la12 = nn.LayerNorm(feats//head)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats//head)
        # self.mlp = nn.Sequential(
        #     GroupedLinear(feats, mlp_hidden, num_groups = head),
        #     FeatureWiseLinear(head,head),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     GroupedLinear(mlp_hidden, feats, num_groups = head),
        #     FeatureWiseLinear(head,head),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )
        self.mlp = nn.Sequential(
            GroupedLinear(feats, mlp_hidden, num_groups = head),
            nn.GELU(),
            nn.Dropout(dropout),
            GroupedLinear(mlp_hidden, feats, num_groups = head),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def crossvit_head_shuffle(self, x):
        b, n, h, f = x.shape  # (b,n,h,f)

        x = x.permute(0,2,1,3)  # (b,h,n,f)
        
        h1 = x[:,0:1,:,:]   # 1st head (cls+body) in vitTiny (b,1,n-1,f)
        h2 = x[:,1:2,:,:]   # 2nd head
        h3 = x[:,2:3,:,:]   # 3rd head

        
        h1_cls = h1[:,:,0:1,:]  # 1st head cls_tkn (b,1,1,f)
        h1_body = h1[:,:,1:,:]  # 1st head body_tkn (b,1,n-1,f)

        h2_cls = h2[:,:,0:1,:]  # 2nd head ~
        h2_body = h2[:,:,1:,:]

        h3_cls = h3[:,:,0:1,:]
        h3_body = h3[:,:,1:,:]


        # 0. head averages and distributed back
        h_cls_avg = (h1_cls + h2_cls + h3_cls) / 3

        h1=torch.cat([h_cls_avg,h1_body], dim=2)   # (b,1,n,f)
        h2=torch.cat([h_cls_avg,h2_body], dim=2)
        h3=torch.cat([h_cls_avg,h3_body], dim=2)

        # 1. random head shuffle
        # cls_rnd_idx = torch.randperm(3)
        # body_rnd_idx = torch.randperm(3)

        # cls_lst = [h1_cls, h2_cls, h3_cls]
        # body_lst = [h1_body, h2_body, h3_body ]

        # h1 = torch.cat([cls_lst[cls_rnd_idx[0]], body_lst[body_rnd_idx[0]]], dim=2)
        # h2 = torch.cat([cls_lst[cls_rnd_idx[1]], body_lst[body_rnd_idx[1]]], dim=2)
        # h3 = torch.cat([cls_lst[cls_rnd_idx[2]], body_lst[body_rnd_idx[2]]], dim=2)

        # breakpoint()

        # # 2. ordered head shuffle
        # h1=torch.cat([h1_cls,h2_body], dim=2)   # (b,1,n,f)
        # h2=torch.cat([h2_cls,h3_body],dim=2)
        # h3=torch.cat([h3_cls,h1_body], dim=2)





        out = torch.cat([h1,h2,h3], dim=1)  # (b,h,n,f)
        out = out.permute(0,2,1,3)  # (b,n,h,f)
        
        # print('head shuffled ordered')
        # breakpoint()

        return out 
        
    def forward(self, x):
        # breakpoint()
        nvtx.range_push('model forward_split')
        b, n, f = x.size()
        x = x.view(b, n, self.head, self.feats//self.head)
        out = self.msa(self.la1(x)) + x


        out = self.crossvit_head_shuffle(out)

        out = self.msa(self.la12(x)) + x    # 추가 another self attn layer 


        # breakpoint()
        out = self.mlp(self.la2(out)) + out
        nvtx.range_pop()
        return out.flatten(2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        self.k = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        self.v = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        
        self.o = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        
        # JINLOVESPHO 
        self.head_shuffle = FeatureWiseLinear(head,head)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #batch, seq_len, dim///
        # breakpoint()
        b, n, h, f = x.size()

        q = self.q(x).transpose(1,2)    # (b,h,n,f)
        k = self.k(x).transpose(1,2)
        v = self.v(x).transpose(1,2)

        # nvtx.range_push('Attention + score')
        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        
        # JINLOVESPHO (head shuffle before attn)
        # attn = self.head_shuffle(attn)  # (b,n,h,f)
        
        # nvtx.range_pop()
        o = self.dropout(self.o(attn))
        
        
        # JINLOVESPHO (head shuffle after attn)
        o = self.head_shuffle(o)  # (b,n,h,f)
        
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
        # x = (.., h, f//h)
        # Apply each linear layer to its corresponding group
        # breakpoint()
        out = torch.einsum("...gi, gij->...gj", x, self.weight)
        # out2 = torch.einsum("...i,...ij->...j",x, self.weight)    # 위와 동일.
        if self.bias is not None:
            out += self.bias
        return out


class FeatureWiseLinear(nn.Module):
    def __init__(self, in_groups: int, out_groups: int):
        super(FeatureWiseLinear, self).__init__()
        self.linear = nn.Linear(in_groups, out_groups)
    def forward(self, x):
        #b,n,h,f = x.size()
        # breakpoint()
        x = x.transpose(2,3) # b,n,f,h
        x = self.linear(x)
        x = x.transpose(2,3) # b,n,h,f
        return x
        
if __name__=="__main__":
    b,n,h,f = 1, 36, 3, 330
    x = torch.randn(b,n,f).cpu()
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f, f*4, h).cpu()
    torchsummary.summary(net, (n,f), device= 'cpu')
    out = net(x)
    print(out.shape)
    
    flops, params = profile(net, inputs=(x, ))
    print(f'flops: {flops}, params: {params}')






# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         self.feats = feats
#         self.head = head
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats//head)
#         self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats//head)
#         # self.mlp = nn.Sequential(
#         #     GroupedLinear(feats, mlp_hidden, num_groups = head),
#         #     FeatureWiseLinear(head,head),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     GroupedLinear(mlp_hidden, feats, num_groups = head),
#         #     FeatureWiseLinear(head,head),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         # )
#         self.mlp = nn.Sequential(
#             GroupedLinear(feats, mlp_hidden, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             GroupedLinear(mlp_hidden, feats, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
        
#     def forward(self, x):
#         # breakpoint()
#         nvtx.range_push('model forward_split')
#         b, n, f = x.size()
#         x = x.view(b, n, self.head, self.feats//self.head)
#         out = self.msa(self.la1(x)) + x
#         # breakpoint()
#         out = self.mlp(self.la2(out)) + out
#         nvtx.range_pop()
#         return out.flatten(2)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5

#         self.q = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
#         self.k = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
#         self.v = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        
#         self.o = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), )
        
#         # JINLOVESPHO 
#         self.head_shuffle = FeatureWiseLinear(head,head)
        
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         #batch, seq_len, dim///
#         # breakpoint()
#         b, n, h, f = x.size()

#         q = self.q(x).transpose(1,2)    # (b,h,n,f)
#         k = self.k(x).transpose(1,2)
#         v = self.v(x).transpose(1,2)

#         # nvtx.range_push('Attention + score')
#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        
#         # JINLOVESPHO (head shuffle before attn)
#         # attn = self.head_shuffle(attn)  # (b,n,h,f)
        
#         # nvtx.range_pop()
#         o = self.dropout(self.o(attn))
        
        
#         # JINLOVESPHO (head shuffle after attn)
#         o = self.head_shuffle(o)  # (b,n,h,f)
        
#         return o

# from torch import Tensor


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
#         # x = (.., h, f//h)
#         # Apply each linear layer to its corresponding group
#         # breakpoint()
#         out = torch.einsum("...gi, gij->...gj", x, self.weight)
#         # out2 = torch.einsum("...i,...ij->...j",x, self.weight)    # 위와 동일.
#         if self.bias is not None:
#             out += self.bias
#         return out


# class FeatureWiseLinear(nn.Module):
#     def __init__(self, in_groups: int, out_groups: int):
#         super(FeatureWiseLinear, self).__init__()
#         self.linear = nn.Linear(in_groups, out_groups)
#     def forward(self, x):
#         #b,n,h,f = x.size()
#         # breakpoint()
#         x = x.transpose(2,3) # b,n,f,h
#         x = self.linear(x)
#         x = x.transpose(2,3) # b,n,h,f
#         return x
        
# if __name__=="__main__":
#     b,n,h,f = 1, 36, 3, 330
#     x = torch.randn(b,n,f).cpu()
#     # net = MultiHeadSelfAttention(f)
#     net = TransformerEncoder(f, f*4, h).cpu()
#     torchsummary.summary(net, (n,f), device= 'cpu')
#     out = net(x)
#     print(out.shape)
    
#     flops, params = profile(net, inputs=(x, ))
#     print(f'flops: {flops}, params: {params}')



