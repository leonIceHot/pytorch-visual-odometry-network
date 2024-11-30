import torch
import torch.nn as nn

def elu_feature_map(x):
    return nn.functional.elu(x) + 1.0

class LinearAttentionX(nn.Module):
    def __init__(self, d_model, nhead, eps=1e-6):
        super(LinearAttentionX, self).__init__()
        
        self.feature_map = elu_feature_map
        self.eps         = eps
        self.nhead       = nhead
        self.mid_dim     = d_model // nhead   # d_model = head * self.mid_dim
        self.query_conv  = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.key_conv    = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.value_conv  = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
    
    def forward(self, x):
        bs, C, height, width = x.size()      # C = self.nhead * self.mid_dim
        # Query
        queries = self.query_conv(x).view(bs, -1, height*width).permute(0, 2, 1)
        queries = queries.view(bs, height*width, self.nhead, self.mid_dim)  #[N, L, H, D]
        # Key
        keys    = self.key_conv(x).view(bs, -1, height*width).permute(0, 2, 1)
        keys    = keys.view(bs, height*width, self.nhead, self.mid_dim)     #[N, S, H, D]
        # Value
        values  = self.value_conv(x).view(bs, -1, height*width).permute(0, 2, 1)
        values  = values.view(bs, height*width, self.nhead, self.mid_dim)   #[N, S, H, D]
        
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        
        KV = torch.einsum("nshd,nshv->nhdv", K, values) # (S,D)' @ (S,V)
        
        Z = (1. / torch.einsum("nlhd,nhd->nlh", Q.float(), K.float().sum(dim=1))+self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q.float(), KV.float(), Z.float())
        queried_values = queried_values.reshape(bs, height*width, -1).permute(0,2,1)
        queried_values = queried_values.view(bs, -1, height, width).contiguous()
        
        return queried_values


# Test codes
bs       = 8
channels = 256
height   = 32
width    = 32
nhead    = 8
x   = torch.rand((bs, channels, height, width))
# print("x.shape = {}".format(x.shape))      # x:[8, 256, 32, 32]
LA  = LinearAttentionX(d_model=channels, nhead=nhead)
out = LA(x) # 相當於 out = LA.forward(x)
# print("out.shape = {}".format(out.shape))  # out:[8, 256, 32, 32]

