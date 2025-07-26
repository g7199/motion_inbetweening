import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from tqdm import tqdm

# --- 기존에 사용하던 helper 함수들을 그대로 사용 ---
from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
from data_loaders.data_sampler import get_data

def get_bvh_frame_count(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if "Frames:" in line:
                return int(line.strip().split()[1])
    return 0

class MotionClipDataset(Dataset):
    def __init__(self, bvh_dir, clip_length=180, feat_bias=5.0):
        self.clip_length = clip_length
        self.bvh_files = sorted(glob.glob(f"{bvh_dir}/**/*.bvh", recursive=True))
        print(f"Found {len(self.bvh_files)} BVH files in {bvh_dir}")
        
        self.mean = np.load('data/mean.npy')
        std = np.load('data/std.npy')
        std += 1e-8
        std[:4] /= feat_bias
        self.std = std

        self.motion_cache = {}

        print("Scanning files and creating clip index...")
        num_clips_per_file = []
        for file_path in tqdm(self.bvh_files):
            num_frames = get_bvh_frame_count(file_path)
            if num_frames >= clip_length:
                num_clips_per_file.append(num_frames - clip_length + 1)
            else:
                num_clips_per_file.append(0)
        
        self.cum_clips_per_file = np.cumsum(num_clips_per_file)
        self.total_clips = self.cum_clips_per_file[-1] if len(self.cum_clips_per_file) > 0 else 0

        print(f"Total {self.total_clips} trainable clips found.")

    def __len__(self):
        return self.total_clips

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_clips_per_file, idx, side='right')
        start_frame = idx - self.cum_clips_per_file[file_idx - 1] if file_idx > 0 else idx

        if file_idx not in self.motion_cache:

            file_path = self.bvh_files[file_idx]
            root, motion = parse_bvh(file_path)
            
            joint_order = get_preorder_joint_list(root)
            motion.build_quaternion_frames(joint_order)
            virtual_root = motion.apply_virtual(root)
            self.motion_cache[file_idx] = (virtual_root, motion)

        virtual_root, motion = self.motion_cache[file_idx]
        

        clip_data = get_data(motion, virtual_root, self.clip_length, start_frame)
        clip_data = (clip_data - self.mean) / self.std
        return torch.FloatTensor(clip_data)