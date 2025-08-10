import torch
import torch.nn as nn
import pdb

from util import get_clones

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        
    def forward(self, x, cond_emb):
        x2 = self.self_attn(x, x, x)[0]
        x = self.norm1(x + x2)
        x = self.act1(x)
        x2 = self.cross_attn(x, cond_emb, cond_emb)[0]
        x = self.norm2(x + x2)
        x = self.act2(x)
        x2 = self.ff(x)
        x = self.norm3(x + x2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, cond_emb):
        for i in range(self.N):
            cond_emb = self.layers[i](x, cond_emb)
        return self.act(cond_emb)
    
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, max_len, d_model):
#         super.__init__()
#         self.pos_embedding = nn.Embedding(max_len, d_model)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         positions = torch.arange(0, seq_len, device=x.device).unsqueeze()

class CondPINN(nn.Module):
    def __init__(self, cond_len, d_out, d_model, d_hidden, d_ff, N, heads):
        super(CondPINN, self).__init__()

        self.xt_input_projection = nn.Linear(2, d_model)
        self.cond_input_projection = nn.Linear(cond_len, d_model)

        self.ffn_encoder = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

        self.attn_encoder = Encoder(d_model, N, heads)

        self.fusion = nn.Sequential(*[
            nn.Linear(2*d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out),
        ])
    
    def forward(self, x, t, conds):
        xt = torch.cat((x, t), dim=-1)
        print(f"xt shape: {xt.shape}")
        cond_all = torch.cat(conds, dim=-1)
        print(f"cond_all shape: {cond_all.shape}")

        xt_emb = self.xt_input_projection(xt)
        print(f"xt_emb shape: {xt_emb.shape}")
        cond_emb = self.cond_input_projection(cond_all)
        cond_emb = cond_emb.squeeze(0)

        ffn_out = self.ffn_encoder(xt_emb)
        attn_out = self.attn_encoder(xt_emb, cond_emb)

        fusion_input = torch.cat((ffn_out, attn_out), dim=-1)
        output = self.fusion(fusion_input)

        return output