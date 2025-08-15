import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_tools.bvh_controller import Motion, MotionFrame, get_preorder_joint_list, parse_bvh, VirtualRootJoint
from pyglm import glm
import torch
import torch.nn.functional as F

def convert_6d_to_rotmat_np(d6: np.ndarray) -> np.ndarray:
    """Numpy를 사용하여 6D 회전 표현을 3x3 회전 행렬로 변환합니다."""
    a1 = d6[:3]
    a2 = d6[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def convert_6d_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    """Torch를 사용하여 6D 회전 표현을 3x3 회전 행렬로 변환합니다."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1  # dot 대신 sum for batch
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1) 

def parse_bvh_skeleton(template_path: str):
    """
    BVH 파일에서 골격 정보(오프셋, End Site 여부)를 파싱합니다.
    """
    with open(template_path, 'r') as f:
        lines = f.readlines()

    joint_offsets = []
    is_end_site = [] # 각 관절이 End Site인지 여부를 저장

    for line in lines:
        line = line.strip()
        
        if "ROOT" in line or "JOINT" in line:
            is_end_site.append(False)
        elif "End Site" in line:
            is_end_site.append(True)
            
        if "OFFSET" in line:
            parts = line.split()[1:]
            joint_offsets.append(np.array([float(p) for p in parts]))
        elif "MOTION" in line:
            break

    print(len(is_end_site))
            
    return np.array(joint_offsets), is_end_site

def tensor_to_kinematics(motion_tensor, template_path):
    print(motion_tensor.shape)
    print("Parsing skeleton hierarchy (including End Sites)...")
    joint_offsets, is_end_site = parse_bvh_skeleton(template_path)
    root, motion = parse_bvh(template_path)
    root = VirtualRootJoint(root)
    num_total_joints = len(is_end_site)
    
    num_frames = motion_tensor.shape[0]
    all_frames_data = []

    current_global_pos = np.array([0.0, 0.0, 0.0])
    current_local_pos = np.array([0.0, 0.0, 0.0])
    current_yaw_angle_rad = 0.0

    
    for frame_idx in range(num_frames):
        frame_data = motion_tensor[frame_idx]

        linear_velocity_xz = frame_data[0:2]
        angular_velocity_yaw_rad = frame_data[2]
        root_height_y = frame_data[3]
        root_local_rot_6d = frame_data[4:10]
        joint_rot_6d_flat = frame_data[10:]

        current_global_pos[0] += linear_velocity_xz[0]
        current_global_pos[2] += linear_velocity_xz[1]
        current_local_pos[1] = root_height_y 
        current_yaw_angle_rad += angular_velocity_yaw_rad
        c, s = np.cos(current_yaw_angle_rad), np.sin(current_yaw_angle_rad)
        global_yaw_rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        global_transform = np.identity(4)
        global_transform[:3, :3] = global_yaw_rot_mat
        global_transform[:3, 3] = current_global_pos

        root_local_rot_mat = convert_6d_to_rotmat_np(root_local_rot_6d)
        root_local_transform = np.identity(4)
        root_local_transform[:3, :3] = root_local_rot_mat
        root_local_transform[:3, 3] = current_local_pos

        joint_local_transforms = []
        rot_data_idx = 0 

        for i in range(1, num_total_joints):
            local_transform = np.identity(4)
            
            if is_end_site[i]:
                # 이 노드가 End Site일 경우: 회전은 Identity, 위치는 오프셋
                local_transform[:3, 3] = joint_offsets[i]
            else:
                # 일반 JOINT일 경우: 회전은 텐서에서, 위치는 오프셋
                start = rot_data_idx * 6
                end = start + 6
                joint_6d = joint_rot_6d_flat[start:end]
                
                local_rot_mat = convert_6d_to_rotmat_np(joint_6d)
                local_transform[:3, :3] = local_rot_mat
                local_transform[:3, 3] = joint_offsets[i]
                
                # 다음 회전 데이터를 가리키도록 인덱서 증가
                rot_data_idx += 1
                
            joint_local_transforms.append(local_transform)

        current_frame_data = np.stack([global_transform, root_local_transform] + joint_local_transforms)
        all_frames_data.append(current_frame_data)

    return root, all_frames_data

def populate_kinematics_dfs(root, current_frame_data):

    matrix_iterator = iter(current_frame_data)

    def _assign_dfs(joint):
        try:
            joint.kinematics = glm.mat4(next(matrix_iterator))
        except StopIteration:
            print("Warning: Not enough matrices in current_frame_data to populate all joints.")
            return

        for child in joint.children:
            _assign_dfs(child)

    _assign_dfs(root)
    
