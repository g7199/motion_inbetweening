import torch
import os
import argparse
from tqdm import tqdm
import numpy as np
import yaml
from types import SimpleNamespace
import glob
from pyglm import glm

# BVH 처리 관련
from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
from data_loaders.data_sampler import get_data

# 모델 및 추론 관련
from diffusion.diffusion import Diffusion
from models.model import MotionTransformer
from data_loaders.data_loader import MotionClipDataset
from bvh_tools.reverse import tensor_to_kinematics
from utils.encode import encode
from utils.inference_tools import frames_to_bvh, get_joint_positions_dfs
from utils.keyframe import KeyframeSelector

selector = KeyframeSelector(ratio=0.1, mode='uniform')

def extract_all_frames_positions(motion, virtual_root, max_frames=None, is_original=False, start_frame=0):
    """이미 계산된 motion과 virtual_root로 모든 프레임의 joint positions 추출"""
    
    # 전체 프레임 수 확인
    total_frames = motion.frames
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    print(f"Total frames to process: {total_frames}")
    
    all_frames_positions = []
    first_kin = None  # is_original=True일 때 첫 번째 프레임 기준으로 상대화
    
    for frame_idx in tqdm(range(total_frames), desc="Extracting positions"):

        # 해당 프레임 설정
        if is_original:
            frame_idx += start_frame
        motion.apply_to_skeleton(frame_idx, virtual_root)
        
        if is_original:
            # 첫 번째 코드와 동일한 상대화 과정 적용
            global_kinematics = np.array(virtual_root.kinematics)
            
            if first_kin is None:
                # 첫 번째 프레임을 기준으로 저장
                first_kin = global_kinematics.copy()
                virtual_root.kinematics = glm.mat4(1.0)
            else:
                # 첫 번째 프레임 기준으로 상대화
                first_inv = np.linalg.inv(first_kin)
                relative_kinematics = first_inv @ global_kinematics
                
                # GLM으로 적용
                virtual_root.kinematics = glm.mat4(*relative_kinematics.T.flatten())
        
        # position 리스트 초기화
        position_list = []
        
        # DFS로 position 추출
        get_joint_positions_dfs(virtual_root, position_list)
        
        all_frames_positions.append(np.array(position_list[3:]))
    
    return np.array(all_frames_positions)  # shape: [frames, joints, 3]

def compare_bvh_positions(original_bvh, generated_bvh, max_frames=180, start_frame=0):
    """원본 BVH와 생성된 BVH의 joint positions 비교"""
    print("\n=== BVH Position Comparison ===")
    
    # 원본 BVH positions 추출
    print("Extracting original BVH positions...")
    root_orig, motion_orig = parse_bvh(original_bvh)
    joint_order_orig = get_preorder_joint_list(root_orig)
    motion_orig.build_quaternion_frames(joint_order_orig)
    virtual_root_orig = motion_orig.apply_virtual(root_orig)
    original_positions = extract_all_frames_positions(motion_orig, virtual_root_orig, max_frames, is_original=True, start_frame=start_frame)
    print(original_bvh)
    
    
    # 생성된 BVH positions 추출
    print("Extracting generated BVH positions...")
    root_gen, motion_gen = parse_bvh(generated_bvh)
    joint_order_gen = get_preorder_joint_list(root_gen)
    motion_gen.build_quaternion_frames(joint_order_gen)
    virtual_root_gen = motion_gen.apply_virtual(root_gen)
    generated_positions = extract_all_frames_positions(motion_gen, virtual_root_gen, max_frames, is_original=False, start_frame=0)
    
    print(f"\nOriginal shape: {original_positions.shape}")
    print(f"Generated shape: {generated_positions.shape}")
    
    # 형태가 다르면 경고
    if original_positions.shape != generated_positions.shape:
        print("WARNING: Position array shapes don't match!")
        min_frames = min(original_positions.shape[0], generated_positions.shape[0])
        min_joints = min(original_positions.shape[1], generated_positions.shape[1])
        print(f"Using common dimensions: frames={min_frames}, joints={min_joints}")
        original_positions = original_positions[:min_frames, :min_joints]
        generated_positions = generated_positions[:min_frames, :min_joints]
    
    # 차이 계산
    diff = original_positions - generated_positions  # signed 차이 (먼저 계산)
    position_diff = np.abs(diff)  # 절대 차이 (MAE 기반용)
    squared_diff = diff ** 2  # 제곱 차이 (MSE 기반용)

    # 전체 MSE와 RMSE 계산 (squared_diff 사용으로 중복 제거)
    mse = np.mean(squared_diff)  # 전체 MSE
    rmse = np.sqrt(mse)  # 표준 RMSE: signed diff 기반

    # 통계 출력
    print(f"\nPosition Comparison Statistics:")
    print(f"Mean absolute difference: {np.mean(position_diff):.6f}")
    print(f"Max absolute difference: {np.max(position_diff):.6f}")
    print(f"RMSE: {rmse:.6f}")  # 수정: 표준 RMSE로
    print(f"MSE: {mse:.6f}")

    # 프레임별 MSE (기존 그대로)
    frame_mses = np.mean(squared_diff, axis=(1, 2))  # 각 프레임의 joint(24)와 coord(3) 평균 MSE
    print(f"\nFrame-wise MSEs:")
    print(f"Min frame MSE: {np.min(frame_mses):.6f}")
    print(f"Max frame MSE: {np.max(frame_mses):.6f}")
    print(f"Mean frame MSE: {np.mean(frame_mses):.6f}")  # 참고: 이게 전체 mse와 같음

    # Joint별 MSE (기존 그대로)
    joint_mses = np.mean(squared_diff, axis=(0, 2))  # 각 조인트의 frame(180)와 coord(3) 평균 MSE
    print(f"\nJoint-wise MSEs:")
    for i, mse in enumerate(joint_mses):
        print(f"Joint {i}: {mse:.6f}")
    
    return original_positions, generated_positions, mse

def process_bvh_and_inference(bvh_file_path, config_path='config.yml', start_frame=600, clip_length=180, add_position=False):
    """BVH 파일에서 샘플 생성하고 바로 추론까지 수행"""
    print(f"Processing BVH: {bvh_file_path}")
    
    # config 로드
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    feat_bias = config_dict.get('feat_bias', 1.0)
    posed_ratio = config_dict.get('posed_ratio', 0.1)
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        return d
    
    config = dict_to_namespace(config_dict)
    
    print("Step 1: Creating sample data from BVH...")
    
    # BVH 파싱 및 데이터 추출 (한 번만)
    root, motion = parse_bvh(bvh_file_path)
    joint_order = get_preorder_joint_list(root)
    motion.build_quaternion_frames(joint_order)
    virtual_root = motion.apply_virtual(root)
    clip_data, traj = get_data(motion, virtual_root, time_size=clip_length, start_frame=start_frame)
    
    # mean, std 로드 (훈련 데이터와 동일한 정규화 적용)
    mean = np.load('data/mean_pos.npy')
    std = np.load('data/std_pos.npy')
    std += 1e-8
    std[:4] /= feat_bias
    std[-4:] /= feat_bias
    
    # trajectory 관련 통계
    mean_vel = mean[:3]
    std_vel = std[:3]
    
    # 텐서 변환 및 정규화
    clip_data_tensor = torch.tensor(clip_data, dtype=torch.float32).unsqueeze(0)
    traj_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0)
    clip_data_tensor = (clip_data_tensor - torch.from_numpy(mean).float()) / torch.from_numpy(std).float()
    traj_tensor = (traj_tensor - torch.from_numpy(mean_vel).float()) / torch.from_numpy(std_vel).float()
    
    print(f"Clip data shape: {clip_data_tensor.shape}, dtype: {clip_data_tensor.dtype}")
    print(f"Trajectory shape: {traj_tensor.shape}, dtype: {traj_tensor.dtype}")
    
    # posed_indices 생성
    _, posed_indices = selector.select_keyframes_by_ratio(clip_data_tensor)
    
    print(f"Generated posed_indices: {posed_indices[0]} (total: {len(posed_indices[0])} frames)")
    
    # 샘플 데이터 준비
    sample_data = {
        'clean_motions': clip_data_tensor,
        'conditions': traj_tensor,
        'posed_indices': posed_indices,
        'dataset_stats': {
            'mean': torch.from_numpy(mean).float(),
            'std': torch.from_numpy(std).float(),
            'mean_vel': torch.from_numpy(mean_vel).float(),
            'std_vel': torch.from_numpy(std_vel).float()
        }
    }
    
    print("\nStep 2: Loading model and starting inference...")
    
    # 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
    
    diffusion = Diffusion(num_timesteps=1000, device=device)
    
    if not os.path.isfile(config.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")
    
    print(f"Loading checkpoint: {config.checkpoint}")
    checkpoint = torch.load(config.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint (model state only)")
    
    # 샘플 데이터에서 정보 추출
    feature_dim = sample_data['clean_motions'].shape[-1]
    condition_pos = sample_data['conditions'][0:1]
    clean_motion = sample_data['clean_motions'][0:1]
    posed_indices = sample_data['posed_indices'][0:1]
    posed_data = clean_motion[:, posed_indices[0]]
    
    print(f"Using clip_length: {clip_length}, feature_dim: {feature_dim}")
    print(f"Sample info:")
    print(f"- Condition shape: {condition_pos.shape}")
    print(f"- Posed indices: {posed_indices[0]}")
    print(f"- Posed data shape: {posed_data.shape}")
    
    # 출력 디렉토리 생성
    os.makedirs(config.inference_dir, exist_ok=True)
    if config.is_mp4:
        os.makedirs(config.inference_dir+"/mp4", exist_ok=True)
    os.makedirs(config.inference_dir+"/bvh", exist_ok=True)
    
    # 추론 수행
    generated_bvh_paths = []
    for i in range(config.num_samples):
        print(f"\n--- Generating sample {i+1}/{config.num_samples} ---")
        
        output_path = os.path.join(config.inference_dir, f"mp4/sample_{i+1:03d}.mp4")
        output_bvh_path = os.path.join(config.inference_dir, f"bvh/sample_{i+1:03d}.bvh")
        
        try:
            mse = sample_motion_inference(
                model=model,
                scheduler=diffusion,
                mean=sample_data['dataset_stats']['mean'],
                std=sample_data['dataset_stats']['std'],
                mean_vel=sample_data['dataset_stats']['mean_vel'],
                std_vel=sample_data['dataset_stats']['std_vel'],
                device=device,
                output_path=output_path,
                output_bvh_path=output_bvh_path,
                template_path=config.template_bvh,
                condition_pos=condition_pos,
                clip_length=clip_length,
                feature_dim=feature_dim,
                guidance_scale=config.guidance_scale,
                is_mp4=config.is_mp4,
                posed_data=posed_data,
                posed_indices=posed_indices,
                original_bvh_path=bvh_file_path,
                start_frame=start_frame,
                add_position=add_position
            )
            generated_bvh_paths.append(output_bvh_path)
        
        except Exception as e:
            print(f"✗ Error generating sample {i+1}: {e}")
            continue
    
    print(f"\n🎉 All done! Generated samples saved in: {config.inference_dir}")

def sample_motion_inference(model, scheduler, mean, std, mean_vel, std_vel, device, output_path, output_bvh_path, template_path, 
                         condition_pos, clip_length=180, feature_dim=212, guidance_scale=3.0, is_mp4=True,
                         posed_data=None, posed_indices=None, original_bvh_path=None, start_frame=0, add_position=False):
    """추론용 모션 샘플링 함수"""
    model.eval()
    
    # 랜덤 노이즈로 시작 (float32로 명시적 지정)
    sample = torch.randn((1, clip_length, feature_dim), device=device, dtype=torch.float32)
    condition_pos = condition_pos.to(device, dtype=torch.float32)
    
    # posed_data가 있으면 device로 이동하고 dtype 맞추기
    if posed_data is not None:
        posed_data = posed_data.to(device, dtype=torch.float32)
        
        # posed_indices가 리스트인 경우 처리
        if isinstance(posed_indices, list):
            # 리스트의 첫 번째 요소가 텐서인 경우
            if len(posed_indices) > 0 and isinstance(posed_indices[0], torch.Tensor):
                posed_indices = posed_indices[0].unsqueeze(0).to(device)
            else:
                # 리스트를 텐서로 변환
                posed_indices = torch.tensor(posed_indices, device=device)
        else:
            posed_indices = posed_indices.to(device)
            
        print(f"Using posed constraints at {len(posed_indices[0]) if posed_indices.dim() > 1 else len(posed_indices)} frames")
        print(f"Posed indices: {posed_indices[0] if posed_indices.dim() > 1 else posed_indices}")
        print(f"Posed data shape: {posed_data.shape}")
    
    print(f"Starting denoising process with guidance_scale={guidance_scale}")
    
    with torch.no_grad():
        for t in tqdm(range(scheduler.num_timesteps - 1, -1, -1), desc="Sampling"):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model.cfg_forward(sample, condition_pos, posed_data, posed_indices, t_tensor, guidance_scale=guidance_scale)
            sample = scheduler.p_step(predicted_noise, t_tensor, sample)
            
    # 맨 마지막에 posed constraint 적용
    if posed_data is not None and posed_indices is not None:
        if sample.dtype != posed_data.dtype:
            posed_data = posed_data.to(sample.dtype)
        sample[:, posed_indices[0]] = posed_data
    
    # 생성된 모션을 후처리
    generated_clip = sample.squeeze(0).cpu().numpy()
    
    # mean, std를 numpy로 변환
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    if isinstance(mean_vel, torch.Tensor):
        mean_vel = mean_vel.cpu().numpy()
    if isinstance(std_vel, torch.Tensor):
        std_vel = std_vel.cpu().numpy()
    
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
    
    frames_to_bvh(root, all_frames_data, output_bvh_path, template_bvh=template_path)
    
    # compare에서 positions 계산
    original_positions, generated_positions, mse = compare_bvh_positions(
        original_bvh=original_bvh_path,
        generated_bvh=output_bvh_path,
        max_frames=clip_length,
        start_frame=start_frame
    )

    if not add_position:
        generated_positions = None
        original_positions = None
    
    # 비디오 생성 (compare에서 구한 positions 사용)
    if is_mp4:
        encode(root, all_frames_data, output_filename=output_path, 
               trajectory=trajectory_cloned, traj_mean=mean_vel, traj_std=std_vel,
               positions=generated_positions, original_positions=original_positions)
        print(f"Motion saved to: {output_path}")

    print(f"MSE: {mse:.6f}")
    return mse

def main():
    parser = argparse.ArgumentParser(description="Motion Diffusion BVH Processing and Inference")
    
    parser.add_argument('--bvh_file', type=str, required=True,
                        help='Path to BVH file')
    parser.add_argument('--start_frame', type=int, default=600,
                        help='Start frame for BVH clipping')
    parser.add_argument('--clip_length', type=int, default=180,
                        help='Length of clip in frames')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Config file path')
    parser.add_argument('--compare_only', action='store_true',
                        help='Only compare two BVH files without inference')
    parser.add_argument('--generated_bvh', type=str,
                        help='Path to generated BVH file for comparison')
    parser.add_argument('--add_position', type=bool, default=False,
                        help='Add position')

    args = parser.parse_args()
    
    if args.compare_only and args.generated_bvh:
        # 비교만 수행
        print("=== BVH Comparison Mode ===")
        compare_bvh_positions(args.bvh_file, args.generated_bvh, args.clip_length, start_frame=args.start_frame)
    else:
        # 전체 프로세스 수행
        print("=== Motion Diffusion BVH Processing ===")
        print(f"BVH file: {args.bvh_file}")
        print(f"Start frame: {args.start_frame}")
        print(f"Clip length: {args.clip_length}")
        print(f"Config file: {args.config}")
        print("=" * 38)
        
        process_bvh_and_inference(
            args.bvh_file,
            args.config,
            args.start_frame,
            args.clip_length,
            args.add_position
        )

if __name__ == "__main__":
    main()