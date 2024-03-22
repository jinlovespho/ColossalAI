import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from einops import rearrange, einsum

class TransformerEncoder_Parallel(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder_Parallel, self).__init__()
        # self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention_Parallel(feats, head=head, dropout=dropout)
        # self.la2 = nn.LayerNorm(feats)
        self.bn = nn.BatchNorm2d(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # out = self.msa(self.la1(x)) + x
        # out = self.mlp(self.la2(out)) + out
        out = self.msa(x) + x
        out = self.mlp(out) + out
        return out


class MultiHeadSelfAttention_Parallel(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention_Parallel, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)
        
        self.la1 = nn.LayerNorm(feats//head)
        self.la2 = nn.LayerNorm(feats//head)

    def forward(self, x):
        # breakpoint()
        b, n, d = x.shape
        
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.head)
        key = rearrange(key, 'b n (h d) -> b h n d', h=self.head)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.head)
        
        attn_matrix = einsum(query, key, 'b h n1 d, b h n2 d -> b h n1 n2') #(b,h,n,n)
        score = F.softmax(attn_matrix/self.sqrt_d, dim=-1)  #(b,h,n,n)
        attn = einsum(score, value, 'b h n1 n2, b h n2 d -> b n1 h d')  # (b n h d)
        
        o = self.dropout(self.o(attn.flatten(2)))
        
        return o


if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder_Parallel(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



