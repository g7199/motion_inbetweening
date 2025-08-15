import torch, math

def sinusoidal_encoding(positions, d_model):
    positions = positions.float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    div_term = div_term.to(positions.device)
    
    pe = torch.zeros(positions.shape[0], d_model, device=positions.device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    
    return pe