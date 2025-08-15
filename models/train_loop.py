import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from data_loaders.data_loader import MotionClipDataset
from diffusion.diffusion import Diffusion
from models.model import MotionTransformer
from models.sample_motions import sample_motion_while_training
import wandb

def train(args):
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Training {args.num_epochs} epochs")

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

    # WandB 초기화 (resume 지원 추가)
    wandb_run_id = None
    if args.resume:
        wandb.init(
            project="noise",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True),
            resume="allow"  # resume 지원 (0.21.0 호환)
        )
    else:
        wandb.init(
            project="noise",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True)
        )

    # Resume from checkpoint if provided (호환성 추가: 기존 단순 state_dict 지원)
    start_epoch = args.start_epoch  # --start_epoch로 받은 값 사용
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 기존 형식: 단순 state_dict
            model.load_state_dict(checkpoint)
            print("Loaded legacy checkpoint (model state only). Optimizer/scheduler will be advanced based on --start_epoch.")
            # "계산해서" scheduler advance (lr decay 상태 재현)
            for _ in range(start_epoch):
                scheduler.step()
            start_epoch += 1  # +1로 다음 epoch부터 시작
            
            if wandb_run_id:
                wandb.init(resume="allow", id=wandb_run_id)  # 기존 wandb run 이어짐
            print(f"Resumed at epoch {start_epoch}")
        else:
            print(f"Checkpoint file not found: {args.resume}. Starting from scratch.")

    # main loop
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        epoch_losses = []
        epoch_velocities = []
        epoch_accelerations = []
        for step, clean_motion in enumerate(dataloader):
            optimizer.zero_grad()
            
            clean_motion = clean_motion.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (clean_motion.shape[0],), device=device)
            noisy_motion, real_noise = diffusion.add_noise(clean_motion, t)

            predicted_noise = model(noisy_motion, t)
            noise_loss = F.mse_loss(predicted_noise, real_noise)

            loss = noise_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f} (Noise only)")
        
        # 에폭마다 모델 저장
        if (epoch - start_epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:  # resume 후 간격 맞춤
            save_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # lr decay 상태 저장
                'wandb_run_id': wandb.run.id  # wandb run ID 저장 for resume
            }, save_path)
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
            

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({"epoch_loss": mean_loss, "epoch": epoch}, step=epoch)

    wandb.finish()