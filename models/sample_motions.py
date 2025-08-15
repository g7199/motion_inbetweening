import torch
from tqdm import tqdm
from bvh_tools.reverse import tensor_to_kinematics
from utils.encode import encode

def sample_motion_while_training(model, scheduler, mean, std, device, output_path, template_path, clip_length=180, feature_dim=211):
    model.eval()

    sample = torch.randn((1, clip_length, feature_dim), device=device)

    for t in tqdm(range(scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
        with torch.no_grad():
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model(sample, t_tensor)
            sample = scheduler.p_step(predicted_noise, t_tensor, sample)

    generated_clip = sample.squeeze(0).cpu().numpy()
    
    denormalized_clip = generated_clip * std + mean
    denormalized_clip = denormalized_clip[:, :142]
    root, all_frames_data = tensor_to_kinematics(denormalized_clip, template_path=template_path)
    print(f"Generated motion with {len(all_frames_data)} frames.")
    encode(root, all_frames_data, output_filename=output_path)

    model.train()