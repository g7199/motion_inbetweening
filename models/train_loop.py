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
    sample_conditions = []
    samples_collected = False
    max_samples = 10

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
        ff_size=args.ff_size, nhead=args.nhead, dropout=args.dropout, activation=args.activation, uncond_prob=args.uncond_prob
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_anneal_steps)

    # WandB 초기화 - resume 로직 수정
    wandb_run_id = None
    start_epoch = args.start_epoch
    
    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 새로운 형식: dict with multiple keys
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                wandb_run_id = checkpoint.get('wandb_run_id', None)
                print(f"Loaded full checkpoint. Resuming from epoch {start_epoch}")
            else:
                # 기존 형식: 단순 state_dict
                model.load_state_dict(checkpoint)
                print("Loaded legacy checkpoint (model state only). Optimizer/scheduler will be advanced based on --start_epoch.")
                # scheduler advance (lr decay 상태 재현)
                for _ in range(start_epoch):
                    scheduler.step()
                start_epoch += 1
            
        else:
            print(f"Checkpoint file not found: {args.resume}. Starting from scratch.")
            args.resume = None  # resume 플래그 해제

    # WandB 초기화 (resume 여부에 따라)
    if args.resume and wandb_run_id:
        wandb.init(
            project="noise",
            id=wandb_run_id,
            resume="allow",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True)
        )
        print(f"Resumed wandb run: {wandb_run_id}")
    else:
        wandb.init(
            project="noise",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True)
        )

    # main loop
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        epoch_losses = []

        # DataLoader unpacking 수정
        for step, (clean_motion, condition_pos) in enumerate(dataloader):
            optimizer.zero_grad()
            
            clean_motion = clean_motion.to(device)
            condition_pos = condition_pos.to(device)

            # 첫 번째 에폭의 첫 번째 배치에서만 샘플 수집
            if not samples_collected and step == 0 and epoch == start_epoch:
                num_to_save = min(max_samples, condition_pos.shape[0])
                for i in range(num_to_save):
                    # torch.FloatTensor로 변환 확실히 하기
                    sample_cond = condition_pos[i:i+1].clone().cpu()
                    if not isinstance(sample_cond, torch.Tensor):
                        sample_cond = torch.FloatTensor(sample_cond)
                    sample_conditions.append(sample_cond)
                
                samples_collected = True
                print(f"Collected {num_to_save} sample conditions from first batch")
                
                # 바로 저장
                torch.save(sample_conditions, os.path.join(samples_dir, 'sample_conditions_for_reproduction.pt'))

            t = torch.randint(0, diffusion.num_timesteps, (clean_motion.shape[0],), device=device)
            noisy_motion, real_noise = diffusion.add_noise(clean_motion, t)

            predicted_noise = model(noisy_motion, condition_pos, t)
            noise_loss = F.mse_loss(predicted_noise, real_noise)

            loss = noise_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f} (Noise only)")
        
        # 에폭마다 모델 저장
        if (epoch - start_epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wandb_run_id': wandb.run.id  # 현재 wandb run ID 저장
            }, save_path)
            print(f"Epoch {epoch} model saved to {save_path}")

            # 샘플 생성 - sample_conditions가 수집된 경우에만
            if sample_conditions:
                epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch}")
                print(f"\n--- Epoch {epoch}: Generating sample motions ---")
                os.makedirs(epoch_samples_dir, exist_ok=True)
                
                for i in range(min(max_samples, len(sample_conditions))):
                    sample_output_path = os.path.join(epoch_samples_dir, f"{i}.mp4")
                    try:
                        sample_motion_while_training(
                            model=model, scheduler=diffusion, 
                            mean=dataset.mean, std=dataset.std,
                            output_path=sample_output_path,
                            template_path=args.template_bvh,
                            condition_pos=sample_conditions[i],
                            device=device,
                            clip_length=args.clip_length,
                            feature_dim=args.feature_dim,
                            guidance_scale=args.guidance_scale
                        )
                    except Exception as e:
                        print(f"Error generating sample {i}: {e}")
                        continue
                        
                print(f"Epoch {epoch} samples saved in '{epoch_samples_dir}'")

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({"epoch_loss": mean_loss, "epoch": epoch}, step=epoch)

    wandb.finish()