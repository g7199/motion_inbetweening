import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from data_loaders.data_loader import MotionClipDataset
from diffusion.diffusion import Diffusion
from models.model import MotionTransformer
from models.sample_motions import sample_motion_while_training

def train(args):

    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = MotionClipDataset(args.bvh_dir, clip_length=args.clip_length, feat_bias=args.feat_bias)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    diffusion = Diffusion(num_timesteps=1000, device=device)
    model = MotionTransformer(
        feature_dim=args.feature_dim, latent_dim=args.latent_dim, num_layers=args.num_layers, 
        ff_size=args.ff_size, nhead=args.nhead, dropout=args.dropout, activation=args.activation
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_anneal_steps)

    # main loop
    for epoch in tqdm(range(args.num_epochs)):
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

            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f}")
        
        # 에폭마다 모델 저장
        save_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch} model saved to {save_path}")

        epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch}")
        print(f"\n--- Epoch {epoch}: Generating a sample motion ---")
        os.makedirs(epoch_samples_dir, exist_ok=True)
        for i in range(3):
            sample_output_path = os.path.join(epoch_samples_dir, f"{i}.mp4")
            sample_motion_while_training(
                model=model, scheduler=diffusion, 
                mean=dataset.mean, std=dataset.std,
                output_path=sample_output_path,
                template_path=args.template_bvh,
                device=device,
                clip_length=args.clip_length,
                feature_dim=args.feature_dim
            )
        print(f"Epoch {epoch} samples saved in '{epoch_samples_dir}'")