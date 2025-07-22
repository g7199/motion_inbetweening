import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import sinusoidal_encoding

class MotionTransformer(nn.Module):
    def __init__(self, feature_dim=171, latent_dim=256, num_layers=8, ff_size=1024, nhead=4, dropout=0.1, activation="gelu"):
        
        self.feature_dim=feature_dim
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = nhead
        self.dropout = dropout

        self.activation = activation

        #Transformer encoder 사용
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        
        #Projection Layer 정의
        self.input_proj = nn.Linear(self.feature_dim, self.latent_dim)
        self.output_proj = nn.Linear(self.latent_dim, self.feature_dim)

        #Time Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        #pos_encoding
        self.pos_enc = PositionalEncoding(d_model=latent_dim, dropout=dropout)

    def forward(self, x, t):

        x_proj = self.input_proj(x)
        x_pos = self.pos_enc(x_proj)

        time_vector = sinusoidal_encoding(t, self.latent_dim)
        time_embedding = self.time_embed(time_vector)

        x_final = x_pos + time_embedding.unsqueeze(1)

        output = self.seqTransEncoder(x_final)
        predicted_noise = self.output_proj(output)

        return predicted_noise



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = sinusoidal_encoding(torch.arange(max_len), d_model)
        self.register_buffer('pe', pe) # 버퍼로 등록

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)
