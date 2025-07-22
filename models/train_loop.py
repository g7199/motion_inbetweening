import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loaders.data_loader import MotionClipDataset
from diffusion.diffusion import Diffusion
from models.model import MotionTransformer

num_epochs = 100
learning_rate = 1e-4
batch_size = 64

weight_decay = 0.05
lr_anneal_steps = 200000
log_interval = 100
save_interval = 5000

def train(bvh_dir:str, clip_length=180, feat_bias=5.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- 1. 모든 부품 생성 ---
    dataset = MotionClipDataset(bvh_dir, clip_length=clip_length, feat_bias=feat_bias)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffusion = Diffusion(num_timesteps=1000, device=device)
    model = MotionTransformer(feature_dim=171, latent_dim=256, num_layers=8, ff_size=1024, nhead=4, dropout=0.1, activation="gelu").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_anneal_steps)

    for epoch in range(num_epochs):
        for step, clean_motion in enumerate(dataloader):
            optimizer.zero_grad()
            
            clean_motion = clean_motion.to(device)  

            t = torch.randint(0, diffusion.num_timesteps, (clean_motion.shape[0],), device=device)
            noisy_motion, real_noise = diffusion.add_noise(clean_motion, t)

            predicted_noise = model(noisy_motion, t)
            loss = F.mse_loss(predicted_noise, real_noise)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f}")
        
        # 에폭마다 모델 저장
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")