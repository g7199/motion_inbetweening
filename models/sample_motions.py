import torch
from tqdm import tqdm
from bvh_tools.reverse import tensor_to_bvh

def sample_motion_while_training(model, scheduler, mean, std, epoch, device, output_path, template_path, clip_length=180, feature_dim=171):
    print(f"\n--- Epoch {epoch}: Generating a sample motion ---")
    model.eval()

    sample = torch.randn((1, clip_length, feature_dim), device=device)

    for t in tqdm(range(scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
        with torch.no_grad():
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model(sample, t_tensor)
            sample = scheduler.step(predicted_noise, t, sample)

    generated_clip = sample.squeeze(0).cpu().numpy()
    denormalized_clip = generated_clip * std + mean
    tensor_to_bvh(denormalized_clip, output_path=output_path, template_path=template_path)

    model.train()