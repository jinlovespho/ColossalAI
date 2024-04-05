import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from einops import rearrange, einsum

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
            

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        
        # JINLOVESPHO  
        self.tp_feats = feats

        self.q = nn.Linear(feats, self.tp_feats)
        self.k = nn.Linear(feats, self.tp_feats)
        self.v = nn.Linear(feats, self.tp_feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # x: (128,65,384)
        b, n, f = x.size()
        
        # ForkedPdb().set_trace()
        q = self.q(x).view(b, n, self.head, self.tp_feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.tp_feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.tp_feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        
        # MY ATTENTION IMPLEMENTATION
    
        # b, n, d = x.shape
        
        # query = self.q(x)
        # key = self.k(x)
        # value = self.v(x)
        
        # SOL 1
        # query = rearrange(query, 'b n (h d) -> b h n d', h=self.head)
        # key = rearrange(key, 'b n (h d) -> b h n d', h=self.head)
        # value = rearrange(value, 'b n (h d) -> b h n d', h=self.head)    
        
        # # SOL 2
        # # query = query.view(b, n, self.head, -1)
        # # query = query.permute(0, 2, 1, 3)
        # # 나머지 key, value 한테도 똑같이 해주기
              
        # attn_matrix = einsum(query, key, 'b h n1 d, b h n2 d -> b h n1 n2') #(b,h,n,n)
        # score = F.softmax(attn_matrix/self.sqrt_d, dim=-1)  #(b,h,n,n)
        # attn = einsum(score, value, 'b h n1 n2, b h n2 d -> b n1 h d')  # (b n h d) 
        # o = self.dropout(self.o(attn.flatten(2)))
        
        return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



