import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):  # max_len을 config에서 가져오도록 추천
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.embedding.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.embedding.num_embeddings}")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)  # [T]
        pos_embed = self.embedding(positions)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        
        x = x + pos_embed
        return self.dropout(x)

def timestep_embedding(t, dim, max_period=10000):
    # t: [B] Long 또는 float, 반환: [B, dim]
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()
    half = dim // 2
    device = t.device
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half)  # [half]
    args = t.unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, 2*half]
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb  # [B, dim]

class MotionTransformer(nn.Module):
    def __init__(self, feature_dim=211, latent_dim=256, num_layers=8, ff_size=1024, nhead=4, dropout=0.1, activation="gelu", max_len=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = nhead
        self.dropout = dropout
        self.activation = activation
        self.max_len = max_len  # config에서 가져올 수 있음

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        self.input_proj = nn.Linear(self.feature_dim, self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, self.feature_dim)

        # 초기화 (추가: 안정성 위해)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)

        # timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.pos_enc = PositionalEncoding(d_model=self.latent_dim, dropout=self.dropout, max_len=self.max_len + 10)  # + buffer for t_emb 등

    def forward(self, x, t):
        # x: [B, T, F], t: [B] (확산 타임스텝)
        B, T, F = x.shape

        # [T, B, D]
        x = x.permute(1, 0, 2)
        x_emb = self.input_proj(x)          # [T, B, D]

        # timestep embedding
        t = t.to(x_emb.device)
        t_sin = timestep_embedding(t, self.latent_dim)  # [B, D]
        t_emb = self.time_mlp(t_sin)                    # [B, D]

        # Option 1: Concat as first token (기존 방식)
        t_emb = t_emb.unsqueeze(0)  # [1, B, D]
        x_emb = torch.cat([t_emb, x_emb], dim=0)  # [T+1, B, D]

        # Option 2: Broadcast add to each frame (대안: 만약 concat이 안 맞으면 이걸 사용 – 주석 해제)
        # t_emb = t_emb.unsqueeze(0).expand(T, -1, -1)  # [T, B, D]
        # x_emb = x_emb + t_emb  # [T, B, D] (no concat, no h[1:])

        x_emb = self.pos_enc(x_emb)         # [T+1, B, D]   
        h = self.seqTransEncoder(x_emb)     # [T+1, B, D]

        h = h[1:]  # [T, B, D] (첫 번째 token 제거, Option 2 사용 시 이 줄 제거)
        predicted_noise = self.output_proj(h).permute(1, 0, 2)  # [B, T, F]
        return predicted_noise