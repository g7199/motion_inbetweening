import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.embedding.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.embedding.num_embeddings}")

        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos_embed = self.embedding(positions)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        
        x = x + pos_embed
        return self.dropout(x)


def timestep_embedding(t, dim, max_period=10000):
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()
    half = dim // 2
    device = t.device
    freqs = torch.exp(-np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = t.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb


class MotionTransformer(nn.Module):
    def __init__(self, feature_dim=212, latent_dim=256, num_layers=8, ff_size=1024, nhead=4, dropout=0.1, activation="gelu", max_len=512, uncond_prob=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.extended_dim = latent_dim  # === 수정: D+3 대신 latent_dim으로 (nhead 배수 맞춤, 256/4=64)
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = nhead
        self.dropout = dropout
        self.activation = activation
        self.max_len = max_len
        self.uncond_prob = uncond_prob

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.extended_dim,  # 256으로 (배수 맞춤)
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=False
        )
        self.seqTransEncoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # === 수정: input_proj input = 1 + feature_dim + 3 (raw concat 후 proj)
        self.input_proj = nn.Linear(1 + self.feature_dim + 3, self.latent_dim)  # 215 -> 256
        self.output_proj = nn.Linear(self.extended_dim, self.feature_dim)  # 256 -> 212

        # 가중치 초기화
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

        # timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # t_emb를 extended_dim으로 proj (여기선 256 -> 256)
        self.t_proj = nn.Linear(self.latent_dim, self.extended_dim)

        self.pos_enc = PositionalEncoding(d_model=self.extended_dim, dropout=self.dropout, max_len=self.max_len + 1)  # d_model=256

    def forward(self, x, con, posed_data, posed_indices, t, force_unconditional=False):
        # x: [B, T, F], con: [B, T, 3], t: [B]
        B, T, F = x.shape
        device = x.device

        con = con.to(device)
        t = t.to(device)

        if force_unconditional:
            con = torch.zeros_like(con)
        elif self.training:
            mask = torch.rand(B, device=con.device) < self.uncond_prob  # [B]
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(con)  # [B, T, 3]로 expand
            con = con * (~mask).float()  # uncond 배치 con=0

        # posed_data로 해당 인덱스 위치 대체
        x_modified = x.clone()
        if posed_data is not None and posed_indices is not None:
            posed_data = posed_data.to(device)
            # 배치별로 개별 처리 (posed_indices는 [B, num_posed] 형태)
            for b in range(B):
                x_modified[b, posed_indices[b]] = posed_data[b]

        # 포즈가 적용된 위치를 나타내는 마스크 생성
        pose_mask = torch.zeros(B, T, 1, device=device)  # [B, T, 1]
        if posed_indices is not None:
            # 배치별로 마스크 설정
            for b in range(B):
                pose_mask[b, posed_indices[b], 0] = 1.0  # 포즈가 적용된 위치는 1
        
        # x 맨 앞에 포즈 마스크 추가
        x_with_mask = torch.cat([pose_mask, x_modified], dim=2)  # [B, T, 1 + F]

        x_with_mask = x_with_mask.permute(1, 0, 2)  # [T, B, 1 + F]
        con = con.permute(1, 0, 2)  # [T, B, 3]
        x_emb = torch.cat([x_with_mask, con], dim=2)  # [T, B, 1 + F + 3]
        x_emb = self.input_proj(x_emb)  # [T, B, D=256]

        t_sin = timestep_embedding(t, self.latent_dim)
        t_emb = self.time_mlp(t_sin)  # [B, D=256]

        t_emb_expanded = self.t_proj(t_emb)  # [B, 256]

        combined_emb = t_emb_expanded.unsqueeze(0)  # [1, B, 256]
        x_emb = torch.cat([combined_emb, x_emb], dim=0)  # [T+1, B, 256]

        x_emb = self.pos_enc(x_emb)
        h = self.seqTransEncoder(x_emb)

        h = h[1:]  # [T, B, 256]
        predicted_noise = self.output_proj(h).permute(1, 0, 2)  # [B, T, F]
        
        return predicted_noise
    
    def cfg_forward(self, x, con, posed_data, posed_indices, t, guidance_scale=1.0):
        if guidance_scale == 1.0:
            return self.forward(x, con, posed_data, posed_indices, t)
        
        original_training = self.training
        self.eval()  # 항상 eval 모드로 (dropout off, etc.)
        
        if self.training:
            cond_noise = self.forward(x, con, posed_data, posed_indices, t, force_unconditional=False)
            uncond_noise = self.forward(x, con, posed_data, posed_indices, t, force_unconditional=True)
        else:
            with torch.no_grad():
                cond_noise = self.forward(x, con, posed_data, posed_indices, t, force_unconditional=False)
                uncond_noise = self.forward(x, con, posed_data, posed_indices, t, force_unconditional=True)
        
        guided_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
        
        self.train(original_training)
        
        return guided_noise