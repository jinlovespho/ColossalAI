import torch
import torch.nn as nn
import torchsummary
from einops import rearrange 

from networks.layers_parallel import TransformerEncoder_Parallel

class ViT_Parallel(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT_Parallel, self).__init__()
        
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder_Parallel(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x):
        bs, in_c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        # patchify image and embed patches to token
        out = self._to_words(x)
        out = self.emb(out)
        
        # add cls token to each batch
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(bs,1,1), out],dim=1)
            
        # add positional embedding to all tokens(including cls token)
        out = out + self.pos_emb
        
        # Transformer Encoder (Self Attention)
        out = self.enc(out)
        
        # which token to use for classification
        if self.is_cls_token:
            out = out[:,0]          # cls token만 이용하여 classification
        else:
            out = out.mean(dim=1)   # 모든 token(cls 포함) 을 평균내어 classification 에 이용 O
            
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        # Sol 1.
        # out2 = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        # out2 = out2.reshape(x.size(0), self.patch**2 ,-1)
        
        # Sol 2.
        out1 = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', ph=self.patch_size, pw=self.patch_size)
        
        return out1


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViT_Parallel(in_c=c, num_classes= 10, img_size=h, patch=16, dropout=0.1, num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=False)
    # out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c,h,w))
    # print(out.shape)
    