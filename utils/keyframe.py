import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F

class LearnableKeyframeIndexSelector(nn.Module):
    def __init__(self, motion_dim, num_keyframes=5, hidden_dim=256):
        super().__init__()
        self.num_kf = num_keyframes
        self.hidden_dim = hidden_dim
        
        # 각 타임스텝에 스코어 부여
        self.score_net = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 각 프레임에 대해 1개 스코어
        )
        
        # 또는 temporal attention 방식도 가능 (아래에 설명)

    def forward(self, motion, temperature=1.0, hard=True):
        """
        motion: [B, T, D]
        returns:
            selected_indices: [B, K] — long tensor (hard selection, for actual indexing)
            soft_probs: [B, T] — soft probability over time (for gradient flow)
        """
        B, T, D = motion.shape
        K = self.num_kf

        # 각 타임스텝에 대해 스코어 계산
        scores = self.score_net(motion).squeeze(-1)  # [B, T]

        # Gumbel-Softmax 또는 Softmax로 확률화
        probs = F.softmax(scores / temperature, dim=-1)  # [B, T]

        if self.training:
            _, topk_indices = torch.topk(probs, K, dim=-1)  # [B, K] — hard, no grad
            selected_indices = topk_indices
            
            one_hot = F.one_hot(selected_indices, num_classes=T).float()  # [B, K, T]
            one_hot_st = one_hot - probs.unsqueeze(1).detach() + probs.unsqueeze(1)  # [B, K, T]
            key_frames = torch.bmm(one_hot_st, motion)
            
            return key_frames, selected_indices, probs

        else:
            # Inference: just pick top-K
            _, selected_indices = torch.topk(probs, K, dim=-1)
            key_frames = torch.stack([motion[b, selected_indices[b]] for b in range(B)], dim=0)
            return key_frames, selected_indices, probs

class KeyframeSelector:
    def __init__(self, ratio: float, mode: str = 'dp', device: str = 'cuda'):
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between 0.0 and 1.0")
        if mode not in ['dp', 'uniform', 'greedy']:
            raise ValueError("Mode must be 'dp', 'uniform', or 'greedy'")
        
        self.ratio = ratio
        self.mode = mode
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.avg_errors: Optional[List[float]] = None
        self.keyframes_list: Optional[List[torch.Tensor]] = None
        self.motion_data: Optional[torch.Tensor] = None
        self.n_batch: Optional[int] = None
        self.n_frames: Optional[int] = None
        self.n_features: Optional[int] = None
    
    def compute_all_pairs_table_batched(self) -> torch.Tensor:
        """
        Batched computation of all-pairs interpolation error table on GPU for all B items.
        Returns e: (B, T, T) where e[b,i,j] = error from i to j for batch b (i < j).
        """
        B, T, _ = self.motion_data.shape  # Batched data already on GPU
        
        # Initialize error matrix on GPU
        e = torch.zeros(B, T, T, device=self.device)
        
        # Batched over all gaps
        for gap in range(1, T):
            if gap == 1:
                continue  # Zero error
            
            # i_indices for all B (same for all batches)
            i_indices = torch.arange(0, T - gap, device=self.device)  # (num_pairs,)
            j_indices = i_indices + gap  # (num_pairs,)
            num_pairs = len(i_indices)
            
            # Batched start/end poses: (B, num_pairs, F)
            start_poses = self.motion_data[:, i_indices]  # (B, num_pairs, F)
            end_poses = self.motion_data[:, j_indices]   # (B, num_pairs, F)
            
            # Accumulate errors for this gap (batched)
            segment_errors = torch.zeros(B, num_pairs, device=self.device)
            
            for offset in range(1, gap):
                t_indices = i_indices + offset
                alphas = torch.full((B, num_pairs), offset / float(gap), device=self.device)  # (B, num_pairs)
                
                # Interpolate: (B, num_pairs, F)
                interpolated = (1 - alphas.unsqueeze(2)) * start_poses + alphas.unsqueeze(2) * end_poses
                actual = self.motion_data[:, t_indices]  # (B, num_pairs, F)
                
                # Add squared errors: (B, num_pairs)
                segment_errors += torch.sum((interpolated - actual) ** 2, dim=2)
            
            # Take sqrt and assign
            e[:, i_indices, j_indices] = torch.sqrt(segment_errors)
        
        return e
    

    def dp_keyframe_selection(self, batch_idx: int, max_keyframes: int, e_cpu: torch.Tensor) -> Dict[int, Tuple[float, List[int]]]:
        """
        Optimized DP with O(T^2 K) complexity.
        Computes minimum error up to frame i with k keyframes.
        Returns solutions for all m ≤ max_keyframes (total keyframes = m).
        Always includes first and last frames.
        """
        T = self.n_frames
        
        # DP table: dp[i][k] = min error to reach frame i with k keyframes
        # k=1 means only frame 0 is selected as keyframe
        dp = torch.full((T, max_keyframes + 1), float('inf'))
        parent = torch.full((T, max_keyframes + 1), -1, dtype=torch.long)
        
        dp[0, 1] = 0.0  # First frame (frame 0) with 1 keyframe
        
        # Fill DP table (vectorized over j)
        for i in range(1, T):
            for k in range(1, min(i + 2, max_keyframes + 1)):  # k can be up to i+1 (since we can select all frames up to i)
                if k == 1:
                    # Only frame 0 is selected, interpolate from 0 to i
                    dp[i, k] = e_cpu[batch_idx, 0, i]
                    parent[i, k] = 0
                else:
                    # Try all possible previous keyframes j for k-1 keyframes
                    j_range = torch.arange(k-2, i, dtype=torch.long)  # j must be at least k-2 to have k-1 keyframes
                    if len(j_range) == 0:
                        continue
                        
                    prev_errors = dp[j_range, k-1] + e_cpu[batch_idx, j_range, i]
                    
                    valid_mask = prev_errors < float('inf')
                    if valid_mask.any():
                        min_error, min_idx = torch.min(prev_errors[valid_mask], dim=0)
                        dp[i, k] = min_error.item()
                        parent[i, k] = j_range[valid_mask][min_idx.item()]
        
        # Extract solutions for each total_k = 2 to max_keyframes
        solutions = {}
        for total_k in range(2, max_keyframes + 1):
            if dp[T-1, total_k] == float('inf'):
                continue
            
            # Reconstruct keyframes by backtracking
            keyframes = []
            curr_i = T - 1
            curr_k = total_k
            
            # Always include the last frame
            keyframes.append(curr_i)
            curr_i = parent[curr_i, curr_k].item()
            curr_k -= 1
            
            # Backtrack through parent pointers
            while curr_k > 1 and curr_i >= 0:
                keyframes.append(curr_i)
                next_i = parent[curr_i, curr_k].item()
                curr_i = next_i
                curr_k -= 1
            
            # Always include the first frame (frame 0)
            keyframes.append(0)
            keyframes.reverse()
            
            total_error = dp[T-1, total_k].item()
            solutions[total_k] = (total_error, keyframes)
        
        return solutions
    
    def greedy_keyframe_selection(self, batch_idx: int, K: int) -> Tuple[float, torch.Tensor]:
        """Fast greedy algorithm with GPU acceleration."""
        keyframes = [0, self.n_frames - 1]
        motion_batch = self.motion_data[batch_idx]
        
        while len(keyframes) < K:
            max_error = -1
            best_pos = -1
            
            for i in range(len(keyframes) - 1):
                start, end = keyframes[i], keyframes[i + 1]
                if end - start <= 1:
                    continue
                
                start_pose = motion_batch[start]
                end_pose = motion_batch[end]
                
                t_range = torch.arange(start + 1, end, device=self.device)
                alphas = ((t_range - start) / (end - start)).unsqueeze(1)
                interpolated = (1 - alphas) * start_pose + alphas * end_pose
                actual = motion_batch[start + 1:end]
                errors = torch.norm(interpolated - actual, dim=1)
                
                if errors.numel() == 0:
                    continue
                
                segment_max_error, segment_max_idx = torch.max(errors, 0)
                segment_max_error = segment_max_error.item()
                segment_max_idx = segment_max_idx.item()
                
                if segment_max_error > max_error:
                    max_error = segment_max_error
                    best_pos = start + 1 + segment_max_idx
            
            if best_pos == -1:
                break
                
            keyframes.append(best_pos)
            keyframes.sort()
        
        total_error = self.compute_error_for_keyframes_gpu(batch_idx, keyframes)
        return total_error, torch.tensor(keyframes, dtype=torch.long)
    
    def compute_error_for_keyframes_gpu(self, batch_idx: int, keyframes: List[int]) -> float:
        """Compute error for given keyframes using GPU."""
        motion_batch = self.motion_data[batch_idx]
        total_error = 0.0
        
        for i in range(len(keyframes) - 1):
            start = keyframes[i]
            end = keyframes[i + 1]
            
            if end - start <= 1:
                continue
            
            start_pose = motion_batch[start]
            end_pose = motion_batch[end]
            
            t_range = torch.arange(start + 1, end, device=self.device)
            alphas = ((t_range - start) / (end - start)).unsqueeze(1)
            interpolated = (1 - alphas) * start_pose + alphas * end_pose
            actual = motion_batch[start + 1:end]
            
            errors = torch.norm(interpolated - actual, dim=1)
            total_error += errors.sum().item()
        
        return total_error
    
    def select_keyframes_by_ratio(self, motion_data: torch.Tensor) -> Tuple[List[float], List[torch.Tensor]]:
        self.motion_data = motion_data.to(self.device)
        self.shape = motion_data.shape
        
        self.is_batched = len(self.shape) == 3
        if self.is_batched:
            self.n_batch = self.shape[0]
            self.n_frames = self.shape[1]
            self.n_features = self.shape[2]
        else:
            self.n_batch = 1
            self.n_frames = self.shape[0]
            self.n_features = self.shape[1]
            self.motion_data = self.motion_data.unsqueeze(0)
        start_time = time.time()
        
        # Batched all-pairs if dp mode
        e_batched = None
        if self.mode == 'dp':
            e_batched = self.compute_all_pairs_table_batched().cpu()  # (B, T, T) to CPU for DP
        
        self.avg_errors = []
        self.keyframes_list = []
        
        for batch_idx in range(self.n_batch):
            K = max(2, int(self.n_frames * self.ratio))
            
            if self.mode == 'uniform':
                keyframes = torch.linspace(0, self.n_frames - 1, K).round().long().tolist()
                total_error = self.compute_error_for_keyframes_gpu(batch_idx, keyframes)
            elif self.mode == 'greedy':
                total_error, keyframes_list = self.greedy_keyframe_selection(batch_idx, K)
                keyframes = keyframes_list.tolist()
            else:  # 'dp'
                solutions = self.dp_keyframe_selection(batch_idx, K, e_batched)
                # Get closest to K
                if K in solutions:
                    total_error, keyframes = solutions[K]
                else:
                    closest_k = min(solutions.keys(), key=lambda x: abs(x - K))
                    total_error, keyframes = solutions[closest_k]
            
            avg_error = total_error / self.n_frames if self.n_frames > 0 else 0.0
            end_time = time.time()
            
            self.avg_errors.append(avg_error)
            self.keyframes_list.append(torch.tensor(keyframes, dtype=torch.long))
        
        return self.avg_errors, self.keyframes_list

# Create synthetic motion data
def create_synthetic_motion(batch_size=4, n_frames=180, n_features=64, device='cuda'):
    """Create smooth synthetic motion data for testing."""
    t = torch.linspace(0, 4*np.pi, n_frames, device=device)
    motion = torch.zeros(batch_size, n_frames, n_features, device=device)
    
    for b in range(batch_size):
        for f in range(n_features):
            freq = torch.rand(1, device=device) * 2 + 0.5
            phase = torch.rand(1, device=device) * 2 * np.pi
            amplitude = torch.rand(1, device=device) * 2 + 0.5
            motion[b, :, f] = amplitude * torch.sin(freq * t + phase) + 0.1 * torch.randn(n_frames, device=device)
    
    return motion

