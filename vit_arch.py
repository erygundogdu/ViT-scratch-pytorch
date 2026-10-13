import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Embedding(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.patch_size = 16
        self.shape = 640
        self.c = 3
        self.n = (self.shape **2) // (self.patch_size **2)
        self.layer = nn.Linear(self.patch_size**2 * self.c,d)
        self.pos = nn.Parameter(torch.randn(self.n + 1, d))
        self.cls = nn.Parameter(torch.randn(1, d))
    def forward(self,x):
        bs = x.shape[0]
        patches = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size) 
        patches = patches.permute(0, 2, 3, 1, 4,5)
        patches = patches.reshape(bs,self.n,-1) #shape(N,CP^2)
        p_patches = self.layer(patches)
        cls_tokens = self.cls.unsqueeze(0).expand(bs,-1,-1)
        emb = torch.cat([cls_tokens, p_patches],dim= 1) + self.pos
        return emb
        

class SA(nn.Module):
    def __init__(self,d,num_heads):
        super().__init__()
        self.dim = d //num_heads
        self.layer = nn.Linear(d,3*self.dim)
        self.softm = nn.Softmax(dim=-1)
        self.scale = math.sqrt(self.dim)
    def forward(self,x):
        ##x.shape = (N,d)
        qkv = self.layer(x) ##shape N,3d
        q,k,v = qkv.split(self.dim,-1)
        attn = self.softm((q @ k.transpose(-2,-1)) / self.scale)
        out = attn @ v
        return out
    
class MSA(nn.Module):
    def __init__(self,num_heads,d):
        super().__init__()
        self.layer = nn.Linear(d, d)
        self.msa = nn.ModuleList([SA(d,num_heads) for _ in range(num_heads)])
    def forward(self,x):
        lst = []
        for head in self.msa:
            head_out = head(x)
            lst.append(head_out)
        tens = torch.cat(lst,dim=-1) ## N,kDh
        out = self.layer(tens) ## N,D
        return out

class EncoderBlock(nn.Module):
    def __init__(self,d,num_heads):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d)
        self.layernorm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d,d*4), nn.GELU(),nn.Linear(d*4,d))
        self.msa = MSA(num_heads,d)
    def forward(self,x):
         # N,D or N+1 x D
        z = self.msa(self.layernorm1(x)) + x
        z = self.mlp(self.layernorm2(z)) + z
        return z 
class Encoder(nn.Module):
    def __init__(self, l,d,num_heads):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderBlock(d,num_heads) for _ in range(l)])
        self.embedding = Embedding(d)
    def forward(self,x):
        z = self.embedding(x) #N+1,D
        for block in self.encoder:
            z = block(z)
        return z
class Head(nn.Module):
    def __init__(self,d,num_cls):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d,2*d), nn.GELU(),nn.Linear(2*d,num_cls))
        self.norm = nn.LayerNorm(d) 
    def forward(self,x):
        x = self.norm(x)
        z = self.mlp(x)
        return z

class ViTClassifier(nn.Module):
    def __init__(self,l,d,num_heads,cls):
        super().__init__()
        self.encoder = Encoder(l,d,num_heads=num_heads)
        self.head = Head(d,cls)
    def forward(self,x):
        x = self.encoder(x)
        cls_token = x[:,0,:]
        logit = self.head(cls_token)
        return logit
    
img = torch.randn(4,3,640,640)
obj = ViTClassifier(l = 4, d=512,num_heads=8,cls=3)
out = obj(img)
print(out.shape)

## positional encoding N+1,D
## embedding weights c*p**2, d pca
