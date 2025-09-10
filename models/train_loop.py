import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from data_loaders.data_loader import MotionClipDataset
from diffusion.diffusion import Diffusion
from models.model import MotionTransformer
from models.sample_motions import sample_motion_while_training
import wandb
from utils.keyframe import KeyframeSelector

selector = KeyframeSelector(ratio=0.1)

def train(args):
    # 샘플링용 데이터를 한 번에 저장할 딕셔너리
    sample_data_for_reproduction = {}
    samples_collected = False
    max_samples = 10

    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    print(f"Training {args.num_epochs} epochs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    dataset = MotionClipDataset(args.bvh_dir, clip_length=args.clip_length, feat_bias=args.feat_bias, height_threshold=args.height_threshold, velocity_threshold=args.velocity_threshold)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    diffusion = Diffusion(num_timesteps=1000, device=device)
    model = MotionTransformer(
        feature_dim=args.feature_dim, latent_dim=args.latent_dim, num_layers=args.num_layers, 
        ff_size=args.ff_size, nhead=args.nhead, dropout=args.dropout, activation=args.activation, uncond_prob=args.uncond_prob
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)  # === 수정: T_max=epochs (epoch 단위 annealing)

    # WandB 초기화 - resume 로직 수정
    wandb_run_id = None
    start_epoch = args.start_epoch
    traj_mean = dataset.mean_vel
    traj_std = dataset.std_vel
    
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
            project="noise_with_foot_contact",
            id=wandb_run_id,
            resume="allow",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True)
        )
        print(f"Resumed wandb run: {wandb_run_id}")
    else:
        wandb.init(
            project="noise_with_foot_contact",
            settings=wandb.Settings(disable_code=True, disable_git=True, silent=True)
        )

    # main loop
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        epoch_losses = []
        epoch_noise_losses = []
        epoch_contact_losses = []

        # DataLoader unpacking 수정
        for step, (clean_motion, condition_pos) in enumerate(dataloader):

            optimizer.zero_grad()
            clean_motion = clean_motion.to(device)
            condition_pos = condition_pos.to(device)
            B, T, feat_dim = clean_motion.shape

            # === 핵심 수정: 첫 번째 배치에서 모든 샘플링 데이터를 함께 저장 ===
            if not samples_collected and step == 0 and epoch == start_epoch:
                # 현재 배치에서 posed 인덱스 생성 (훈련과 동일한 로직)
                _, current_posed_indices = selector.select_keyframes_by_ratio(clean_motion)

                num_to_save = min(max_samples, condition_pos.shape[0])
                
                sample_data_for_reproduction = {
                    'conditions': condition_pos[:num_to_save].clone().cpu(),
                    'clean_motions': clean_motion[:num_to_save].clone().cpu(),
                    'posed_indices': [idx.clone().cpu() for idx in current_posed_indices[:num_to_save]],  # 리스트의 각 텐서 clone/cpu
                    'dataset_stats': {
                        'mean': dataset.mean,
                        'std': dataset.std,
                        'mean_vel': dataset.mean_vel,
                        'std_vel': dataset.std_vel
                    },
                    'posed_ratio': args.posed_ratio,
                    'clip_length': args.clip_length
                }
                
                samples_collected = True
                print(f"Collected {num_to_save} sample data (NOTE: posed_data uses hints during training)")
                torch.save(sample_data_for_reproduction, 
                           os.path.join(samples_dir, 'sample_data_for_reproduction.pt'))

            # 일반적인 훈련 과정 (매 배치마다 새로운 posed 데이터 생성)
            t = torch.randint(0, diffusion.num_timesteps, (clean_motion.shape[0],), device=device)
            noisy_motion, real_noise = diffusion.add_noise(clean_motion, t)

            _, posed_indices_list = selector.select_keyframes_by_ratio(clean_motion)
            print("hi")
            
            # 리스트를 padded 텐서로 변환 (전부 tensor로 관리)
            max_K = max(len(indices) for indices in posed_indices_list)
            padded_indices = torch.full((B, max_K), -1, dtype=torch.long, device=device)  # pad with -1
            lengths = torch.zeros(B, dtype=torch.long, device=device)
            
            posed_data_padded = torch.zeros(B, max_K, feat_dim, device=device)  # pad with 0
            
            for b in range(B):
                indices = posed_indices_list[b].to(device)
                num_K = len(indices)
                padded_indices[b, :num_K] = indices
                lengths[b] = num_K
                
                posed_data_padded[b, :num_K] = clean_motion[b, indices]

            # model 호출 (padded 텐서 전달) 
            predicted_noise = model(noisy_motion, condition_pos, posed_data_padded, padded_indices, t)

            # === 수정: Motion/Contact loss split (feature_dim=212 가정, 마지막 4-dim=contact)
            non_posed_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            for b in range(B):
                non_posed_mask[b, padded_indices[b, :lengths[b]]] = False

            # 마스크된 loss 계산
            motion_loss = F.mse_loss(
                predicted_noise[:, :, :-4][non_posed_mask], 
                real_noise[:, :, :-4][non_posed_mask]
            )
            contact_loss = F.mse_loss(
                predicted_noise[:, :, -4:][non_posed_mask], 
                real_noise[:, :, -4:][non_posed_mask]
            )

            loss = F.mse_loss(predicted_noise[non_posed_mask], real_noise[non_posed_mask])  # Total loss (weight 조정 가능, e.g., + 0.1 * contact_loss)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_noise_losses.append(motion_loss.item())
            epoch_contact_losses.append(contact_loss.item())

            if step % args.log_interval == 0:
                print(f"Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f} (Motion: {motion_loss.item():.4f}, Contact: {contact_loss.item():.4f})")
        
        # === 수정: scheduler epoch 끝에 step (epoch 단위 annealing)
        scheduler.step()

        # 에폭마다 모델 저장
        if (epoch - start_epoch) % args.save_interval == 0 or (epoch) == args.num_epochs:
            save_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wandb_run_id': wandb.run.id  # 현재 wandb run ID 저장
            }, save_path)
            print(f"Epoch {epoch} model saved to {save_path}")

            # === 수정된 샘플 생성: 저장된 데이터 사용 ===
            if sample_data_for_reproduction:
                epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch}")
                print(f"\n--- Epoch {epoch}: Generating sample motions ---")
                os.makedirs(epoch_samples_dir, exist_ok=True)
                
                for i in range(len(sample_data_for_reproduction['conditions'])):
                    sample_output_path = os.path.join(epoch_samples_dir, f"{i}.mp4")
                    try:
                        # 저장된 데이터에서 개별 샘플 추출
                        sample_condition = sample_data_for_reproduction['conditions'][i:i+1].to(device)
                        sample_clean_motion = sample_data_for_reproduction['clean_motions'][i:i+1].to(device)
                        # posed_indices 처리 - 리스트에서 i번째 요소 추출
                        if isinstance(sample_data_for_reproduction['posed_indices'], list):
                            sample_posed_indices_tensor = sample_data_for_reproduction['posed_indices'][i].to(device)
                            sample_num_K = len(sample_posed_indices_tensor)
                            sample_padded_indices = torch.full((1, sample_num_K), -1, dtype=torch.long, device=device)
                            sample_padded_indices[0, :sample_num_K] = sample_posed_indices_tensor
                        else:
                            sample_padded_indices = sample_data_for_reproduction['posed_indices'][i:i+1].to(device)
                            sample_num_K = sample_padded_indices.shape[1]

                        sample_posed_data = sample_clean_motion[0, sample_posed_indices_tensor[:sample_num_K]]  # 훈련과 동일한 방식!
                        sample_posed_data = sample_posed_data.unsqueeze(0).to(device)  # [1, num_posed, F]
                        
                        sample_motion_while_training(
                            model=model, 
                            scheduler=diffusion, 
                            mean=sample_data_for_reproduction['dataset_stats']['mean'],
                            std=sample_data_for_reproduction['dataset_stats']['std'],
                            output_path=sample_output_path,
                            template_path=args.template_bvh,
                            condition_pos=sample_condition,
                            device=device,
                            clip_length=args.clip_length,
                            feature_dim=args.feature_dim,
                            guidance_scale=args.guidance_scale,
                            traj_mean=sample_data_for_reproduction['dataset_stats']['mean_vel'],
                            traj_std=sample_data_for_reproduction['dataset_stats']['std_vel'],
                            posed_data=sample_posed_data,  # 단일 텐서
                            posed_indices=sample_padded_indices,  # padded 텐서
                        )
                    except Exception as e:
                        print(f"Error generating sample {i}: {e}")
                        continue
                        
                print(f"Epoch {epoch} samples saved in '{epoch_samples_dir}'")

        mean_total_loss = sum(epoch_losses) / len(epoch_losses)
        mean_motion_loss = sum(epoch_noise_losses) / len(epoch_noise_losses)
        mean_contact_loss = sum(epoch_contact_losses) / len(epoch_contact_losses)
        
        wandb.log({
            "epoch_total_loss": mean_total_loss,     # 전체 loss
            "epoch_motion_loss": mean_motion_loss,   # motion 부분만
            "epoch_contact_loss": mean_contact_loss, # contact 부분만
            "learning_rate": scheduler.get_last_lr()[0]  # === 수정: LR log 추가
        }, step=epoch)

        print(f"Loss: {mean_total_loss:.4f} (Motion: {mean_motion_loss:.4f}, Contact: {mean_contact_loss:.4f})")

    wandb.finish()