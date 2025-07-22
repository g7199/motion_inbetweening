from scipy.spatial.transform import Rotation as R
import numpy as np

def rotation_6d_to_euler(d6_rotations):
    # 6D -> 3x3 회전 행렬
    a1, a2 = d6_rotations[..., :3], d6_rotations[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)
    rotation_matrix = np.stack([b1, b2, b3], axis=-1)

    r = R.from_matrix(rotation_matrix)
    return r.as_euler('zyx', degrees=True)


def tensor_to_bvh(motion_tensor, output_path, template_path):

    with open(template_path, 'r') as f:
        lines = f.readlines()
    
    hierarchy_lines = []
    for i, line in enumerate(lines):
        hierarchy_lines.append(line)
        if "MOTION" in line:
            break
            
    num_frames = motion_tensor.shape[0]
    frame_time = 0.016667

    motion_lines = []
    motion_lines.append(f"Frames: {num_frames}\n")
    motion_lines.append(f"Frame Time: {frame_time}\n")

    num_joints = (motion_tensor.shape[1]-3)//3
    for frame_idx in range(num_frames):
        frame_data = motion_tensor[frame_idx]
        
        root_pos = frame_data[0:3]
        root_rot_6d = frame_data[3:9]
        joint_rot_6d = frame_data[9:].reshape((num_joints - 1), 6)
        
        root_rot_euler = rotation_6d_to_euler(root_rot_6d)
        joint_rot_euler = rotation_6d_to_euler(joint_rot_6d)
        
        line_data = [
            f"{root_pos[0]:.6f}", f"{root_pos[1]:.6f}", f"{root_pos[2]:.6f}",
            f"{root_rot_euler[0]:.6f}", f"{root_rot_euler[1]:.6f}", f"{root_rot_euler[2]:.6f}"
        ]
        line_data.extend([f"{angle:.6f}" for euler in joint_rot_euler for angle in euler])
        
        motion_lines.append(" ".join(line_data) + "\n")

    with open(output_path, 'w') as f:
        f.writelines(hierarchy_lines)
        f.writelines(motion_lines)
        
    print(f"BVH file succesfully saved in '{output_path}'")
