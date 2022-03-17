import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTCls(nn.Module):
    def __init__(self,dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0., num_classes=8):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        #
        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        #
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        #self.fc1 = nn.Linear(512, dim)
        self.to_patch_embedding = nn.Linear(512, dim)
        #self.norm2 = nn.LayerNorm(dim)
        self.to_patch_embedding2 = nn.Linear(2304,dim)
        #self.norm1 = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.judge_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, x, audio_feat, target=False, another_x=None, another_audio_feat = None):
        if another_x is not None:
            v_feat = self.to_patch_embedding2(x)
            another_v_feat = self.to_patch_embedding2(another_x)
            b1,n1, _ = v_feat.size()
            audio_feat = self.to_patch_embedding(audio_feat)
            another_audio_feat = self.to_patch_embedding(another_audio_feat)
            b, n, _ = audio_feat.shape
            judge_token = repeat(self.judge_token, '() n d -> b n d', b = b)

            pos1 = self.pos_embedding[:,1:] #target
            v_feat += pos1
            #audio_feat += pos1

            pos2 = self.pos_embedding[:,:1] #source
            another_v_feat += pos2
            #another_audio_feat += pos2

            x = torch.cat((judge_token,v_feat, audio_feat, another_v_feat, another_audio_feat), dim=1)
        else:
            v_feat = self.to_patch_embedding2(x)
            b1,n1, _ = v_feat.size()
            audio_feat = self.to_patch_embedding(audio_feat)
            b, n, _ = audio_feat.shape
            #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            if target is True:
                pos1 = self.pos_embedding[:,1:]
            else:
                pos1 = self.pos_embedding[:,:1]
            pos2 = pos1.repeat(1, v_feat.size()[1], 1)
            v_feat += pos2

            #pos2 = pos1.repeat(1, audio_feat.size()[1], 1)
            #audio_feat += pos2

            x = torch.cat((audio_feat, v_feat,), dim=1)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if another_x is not None:
            return self.mlp_head2(x)
        else:
            return self.mlp_head(x)
