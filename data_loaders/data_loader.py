import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob # 파일 경로를 쉽게 찾기 위해 사용
from tqdm import tqdm

from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
from data_loaders.data_sampler import get_data

class MotionClipDataset(Dataset):
    def __init__(self, bvh_dir, clip_length=180, feat_bias=5.0):
        self.clip_length = clip_length
        self.bvh_files = glob.glob(f"{bvh_dir}/*/*.bvh")
        print(f"Found {len(self.bvh_files)} BVH files in {bvh_dir}")
        self.mean = np.load('data/mean.npy')
        self.std = np.load('data/std.npy')+ 1e-8
        self.std[:3] /= feat_bias

        print("motion data loading")

        frame_lens = []
        for file_path in tqdm(self.bvh_files):
            _, motion = parse_bvh(file_path)
            if motion.frames >= clip_length:
                frame_lens.append(motion.frames - clip_length + 1)
            else:
                frame_lens.append(0)

        total_frames = np.sum(frame_lens)
        self.cum_frame_lens = np.cumsum(frame_lens)
        self.len = total_frames

    def __len__(self):
        return int(self.len)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cum_frame_lens, idx, side='right')

        if file_idx == 0:
            start_frame = idx
        else:
            start_frame = idx - self.cum_frame_lens[file_idx - 1]

        root,motion = parse_bvh(self.bvh_files[file_idx])
        joint_order = get_preorder_joint_list(root)
        motion.build_quaternion_frames(joint_order)
        _ = motion.apply_virtual(root)
        clip_data = get_data(motion, root, self.clip_length, start_frame)
        
        clip_data = (clip_data - self.mean) / self.std
        return torch.FloatTensor(clip_data)
        
