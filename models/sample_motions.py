import torch
from tqdm import tqdm
from bvh_tools.reverse import tensor_to_kinematics
from utils.encode import encode
import numpy as np

def sample_motion_while_training(model, scheduler, mean, std, device, output_path, template_path, condition_pos, traj_mean, traj_std, clip_length=180, feature_dim=212, guidance_scale=3.0, posed_data=None, posed_indices=None ):
    model.eval()
    print("Starting motion sampling...")

    sample = torch.randn((1, clip_length, feature_dim), device=device)
    condition_pos = condition_pos.to(device)  # device 맞춤
    
    with torch.no_grad():
        for t in tqdm(range(scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model.cfg_forward(sample, condition_pos, posed_data, posed_indices, t_tensor, guidance_scale=guidance_scale)
            sample = scheduler.p_step(predicted_noise, t_tensor, sample)

    if posed_data is not None and posed_indices is not None:
        sample[:, posed_indices[0]] = posed_data.to(device)

    generated_clip = sample.squeeze(0).cpu().numpy()
    denormalized_clip = generated_clip * std + mean
    denormalized_clip = denormalized_clip[:, :142]
    root, all_frames_data = tensor_to_kinematics(denormalized_clip, template_path=template_path)
    print(f"Generated motion with {len(all_frames_data)} frames.")

    if isinstance(condition_pos, torch.Tensor):
        trajectory_cloned = condition_pos.clone().detach().cpu().numpy()
    else:
        trajectory_cloned = np.array(condition_pos)  # 리스트나 다른 배열일 경우 numpy로 변환

    if trajectory_cloned.shape[0] == 1:
        trajectory_cloned = trajectory_cloned[0]

    encode(root, all_frames_data, output_filename=output_path, trajectory=trajectory_cloned, traj_mean=traj_mean, traj_std=traj_std)

    model.train()