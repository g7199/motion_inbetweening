import torch
import os
import argparse
from tqdm import tqdm
import numpy as np
import yaml
from types import SimpleNamespace
import math

from diffusion.diffusion import Diffusion
from models.model import MotionTransformer
from data_loaders.data_loader import MotionClipDataset
from bvh_tools.reverse import tensor_to_kinematics
from utils.encode import encode
from utils.inference_tools import frames_to_bvh


def sample_motion_inference(model, scheduler, mean, std, mean_vel, std_vel, device, output_path, output_bvh_path, template_path, 
                          condition_pos, clip_length=180, feature_dim=212, guidance_scale=3.0, is_mp4=True):
    """추론용 모션 샘플링 함수"""
    model.eval()
    
    # 랜덤 노이즈로 시작
    sample = torch.randn((1, clip_length, feature_dim), device=device)
    condition_pos = condition_pos.to(device)
    
    print(f"Starting denoising process with guidance_scale={guidance_scale}")
    
    with torch.no_grad():
        for t in tqdm(range(scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model.cfg_forward(sample, condition_pos, t_tensor, guidance_scale=guidance_scale)
            sample = scheduler.p_step(predicted_noise, t_tensor, sample)
    
    # 생성된 모션을 후처리
    generated_clip = sample.squeeze(0).cpu().numpy()
    
    # 정규화 해제
    denormalized_clip = generated_clip * std + mean
    denormalized_clip = denormalized_clip[:, :142]  # contact 정보 제거
    
    # BVH로 변환
    root, all_frames_data = tensor_to_kinematics(denormalized_clip, template_path=template_path)
    print(f"Generated motion with {len(all_frames_data)} frames.")
    
    # 조건 처리
    if isinstance(condition_pos, torch.Tensor):
        trajectory_cloned = condition_pos.clone().detach().cpu().numpy()
    else:
        trajectory_cloned = np.array(condition_pos)
    
    if trajectory_cloned.shape[0] == 1:
        trajectory_cloned = trajectory_cloned[0]
    
    # 비디오 생성
    if is_mp4:
        encode(root, all_frames_data, output_filename=output_path, trajectory=trajectory_cloned, traj_mean=mean_vel, traj_std=std_vel)
        print(f"Motion saved to: {output_path}")

    frames_to_bvh(root, all_frames_data, output_bvh_path, template_bvh=template_path)


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # dict를 SimpleNamespace로 변환 (args처럼 사용하기 위해)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        return d
    
    return dict_to_namespace(config)


def load_model_and_dataset(config):
    """모델과 데이터셋 로드"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 데이터셋 로드 (정규화 파라미터를 위해)
    dataset = MotionClipDataset(
        config.bvh_dir, 
        clip_length=config.clip_length, 
        feat_bias=config.feat_bias, 
        height_threshold=config.height_threshold, 
        velocity_threshold=config.velocity_threshold
    )
    
    # 모델 초기화
    model = MotionTransformer(
        feature_dim=config.feature_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        ff_size=config.ff_size,
        nhead=config.nhead,
        dropout=config.dropout,
        activation=config.activation,
        uncond_prob=config.uncond_prob
    ).to(device)
    
    # 디퓨전 스케줄러 초기화
    diffusion = Diffusion(num_timesteps=1000, device=device)
    
    # 체크포인트 로드
    if not os.path.isfile(config.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    
    print(f"Loading checkpoint: {config.checkpoint}")
    checkpoint = torch.load(config.checkpoint, map_location=device)
    
    # 체크포인트 형식 확인 후 로드
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        # 기존 형식: 단순 state_dict
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint (model state only)")
    
    return model, diffusion, dataset, device


def generate_random_trajectory(clip_length, trajectory_type="straight"):
    """랜덤 궤적을 속도로 생성 (yaw_vel 포함)"""
    if trajectory_type == "straight":
        # 직선 궤적
        start_x, start_z = 0.0, 0.0
        end_x, end_z = np.random.uniform(-5, 5), np.random.uniform(5, 10)
        x_coords = np.linspace(start_x, end_x, clip_length)
        z_coords = np.linspace(start_z, end_z, clip_length)
        
        # 직선이므로 yaw는 일정한 방향
        target_yaw = math.atan2(end_x - start_x, end_z - start_z)  # 목표 방향
        yaw_coords = np.full(clip_length, target_yaw)  # 일정한 각도
    
    elif trajectory_type == "circle":
        # 원형 궤적
        radius = np.random.uniform(2, 5)
        center_x, center_z = 0.0, radius
        angles = np.linspace(0, 2*np.pi, clip_length)
        x_coords = center_x + radius * np.cos(angles)
        z_coords = center_z + radius * np.sin(angles)
        
        # 원형 궤적의 yaw: 항상 접선 방향
        yaw_coords = angles + np.pi/2  # 접선 방향 (90도 회전)
    
    elif trajectory_type == "zigzag":
        # 지그재그 궤적
        amplitude = np.random.uniform(2, 4)
        frequency = np.random.uniform(0.02, 0.05)
        z_coords = np.linspace(0, 10, clip_length)
        x_coords = amplitude * np.sin(frequency * 2 * np.pi * np.arange(clip_length))
        
        # 지그재그의 yaw: 움직임 방향에 따라 계산
        dx = np.gradient(x_coords)  # x 방향 기울기
        dz = np.gradient(z_coords)  # z 방향 기울기
        yaw_coords = np.arctan2(dx, dz)  # 움직임 방향
    
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    # 위치를 속도로 변환
    positions = np.column_stack([x_coords, z_coords])  # [T, 2]
    
    # XZ 속도 계산
    velocities_xz = np.zeros_like(positions)
    velocities_xz[:-1] = positions[1:] - positions[:-1]
    velocities_xz[-1] = velocities_xz[-2]  # 마지막 프레임
    
    # Yaw 속도 계산
    yaw_velocities = np.zeros(clip_length)
    yaw_velocities[:-1] = yaw_coords[1:] - yaw_coords[:-1]
    
    # Yaw 속도에서 각도 wrap-around 처리
    yaw_velocities = np.where(yaw_velocities > np.pi, yaw_velocities - 2*np.pi, yaw_velocities)
    yaw_velocities = np.where(yaw_velocities < -np.pi, yaw_velocities + 2*np.pi, yaw_velocities)
    
    yaw_velocities[-1] = yaw_velocities[-2]  # 마지막 프레임
    
    # [T, 3] 형태로 결합: [vel_x, vel_z, yaw_vel]
    velocities_3d = np.column_stack([velocities_xz[:, 0], velocities_xz[:, 1], yaw_velocities])
    
    return torch.FloatTensor(velocities_3d*100).unsqueeze(0) 


def inference(config):
    """추론 메인 함수"""
    # 모델과 데이터셋 로드
    model, diffusion, dataset, device = load_model_and_dataset(config)
    
    # 출력 디렉토리 생성 (config의 inference_dir 사용)
    os.makedirs(config.inference_dir, exist_ok=True)
    if config.is_mp4:
        os.makedirs(config.inference_dir+"/mp4", exist_ok=True)
    os.makedirs(config.inference_dir+"/bvh", exist_ok=True)
    
    for i in range(config.num_samples):
        print(f"\n--- Generating sample {i+1}/{config.num_samples} ---")
        
        # 조건 생성
        if hasattr(config, 'condition_file') and config.condition_file:
            print(f"Loading condition from: {config.condition_file}")
            condition_pos = torch.load(config.condition_file, map_location='cpu')

            # 리스트면 하나 뽑기
            if isinstance(condition_pos, list):
                condition_pos = condition_pos[i % len(condition_pos)]

            # 텐서 강제 캐스팅
            if isinstance(condition_pos, np.ndarray):
                condition_pos = torch.from_numpy(condition_pos)

            # (T,3) -> (1,T,3)
            if condition_pos.dim() == 2:
                condition_pos = (condition_pos - dataset.mean_vel) / dataset.std_vel
                print(condition_pos)
                condition_pos = condition_pos.unsqueeze(0)

            # dtype, device 정리
            condition_pos = condition_pos.to(dtype=torch.float32)
        else:
            # 랜덤 궤적 생성
            trajectory_types = ["straight", "circle", "zigzag"]
            trajectory_type = np.random.choice(trajectory_types)
            print(f"Generating random {trajectory_type} trajectory")
            condition_pos = generate_random_trajectory(config.clip_length, trajectory_type)
        
        # 샘플 생성
        output_path = os.path.join(config.inference_dir, f"mp4/sample_{i+1:03d}.mp4")
        output_bvh_path = os.path.join(config.inference_dir, f"bvh/sample_{i+1:03d}.bvh")
        
        try:
            sample_motion_inference(
                model=model,
                scheduler=diffusion,
                mean=dataset.mean,
                std=dataset.std,
                mean_vel=dataset.mean_vel,
                std_vel=dataset.std_vel,
                device=device,
                output_path=output_path,
                output_bvh_path=output_bvh_path,
                template_path=config.template_bvh,
                condition_pos=condition_pos,
                clip_length=config.clip_length,
                feature_dim=config.feature_dim,
                guidance_scale=config.guidance_scale,
                is_mp4=config.is_mp4
            )
            print(f"✓ Sample {i+1} generated successfully")
        
        except Exception as e:
            print(f"✗ Error generating sample {i+1}: {e}")
            continue
    
    print(f"\n🎉 Inference completed! Generated samples saved in: {config.inference_dir}")


def main():
    parser = argparse.ArgumentParser(description="Motion Diffusion Inference")
    
    # 필수 파라미터만
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--condition_file', type=str, default=None,
                        help='Path to condition file (.pt). If not provided, random trajectories will be generated')
    parser.add_argument('--mp4', type=str, default=False,
                        help='making mp4')
    
    args = parser.parse_args()
    
    # config 파일 로드
    config = load_config(args.config)
    
    # 명령줄 인자로 필요한 것만 덮어쓰기
    config.checkpoint = args.checkpoint
    config.num_samples = args.num_samples
    config.is_mp4 = args.mp4
    if args.condition_file:
        config.condition_file = args.condition_file
    
    # 파라미터 출력
    print("=== Inference Configuration ===")
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Output dir: {config.inference_dir}")
    print(f"Num samples: {config.num_samples}")
    print(f"Guidance scale: {config.guidance_scale}")
    if hasattr(config, 'condition_file'):
        print(f"Condition file: {config.condition_file}")
    print("=" * 30)
    
    inference(config)


if __name__ == "__main__":
    main()