import numpy as np
import math
import glm
from bvh_tools.reverse import populate_kinematics_dfs

# BVH 조인트 순서
BVH_JOINT_ORDER = [
    'Hips', 'Chest', 'Chest2', 'Chest3', 'Chest4', 'Neck', 'Head',
    'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist',
    'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist',
    'RightHip', 'RightKnee', 'RightAnkle', 'RightToe',
    'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe'
]

def parse_bvh_rotation_orders(template_bvh_path):
    """BVH 파일에서 각 joint의 rotation order를 파싱"""
    joint_rotation_orders = {}
    current_joint = None
    
    with open(template_bvh_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # MOTION 섹션 시작하면 파싱 종료
            if line.startswith('MOTION'):
                break
                
            # Joint 이름 찾기
            if line.startswith('ROOT') or line.startswith('JOINT'):
                parts = line.split()
                if len(parts) >= 2:
                    current_joint = parts[1]
            
            # CHANNELS 라인 찾기
            elif line.startswith('CHANNELS') and current_joint:
                parts = line.split()
                if len(parts) >= 2:
                    num_channels = int(parts[1])
                    channels = parts[2:]
                    
                    # rotation channels만 추출 (Xrotation, Yrotation, Zrotation)
                    rotation_channels = []
                    for channel in channels:
                        if 'rotation' in channel.lower():
                            axis = channel[0].upper()  # X, Y, Z
                            rotation_channels.append(axis)
                    
                    if rotation_channels:
                        joint_rotation_orders[current_joint] = ''.join(rotation_channels)
    
    return joint_rotation_orders

def find_joint_by_name(root, joint_name):
    if hasattr(root, 'name') and root.name == joint_name:
        return root
    if hasattr(root, 'children'):
        for child in root.children:
            result = find_joint_by_name(child, joint_name)
            if result is not None:
                return result
    return None

import math

def quat_to_euler_xyz(q):
    """Quaternion → Euler (XYZ 고정)"""
    r_x = -math.asin(max(-1, min(1, -2*(q.y*q.z - q.w*q.x))))
    r_y = -math.atan2(2*(q.x*q.z + q.w*q.y), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
    r_z = -math.atan2(2*(q.x*q.y + q.w*q.z), q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z)
    return [r_x, r_y, r_z]

def quat_to_euler_by_order(q, order='ZYX'):
    """Quaternion → Euler (주어진 순서)"""
    base = quat_to_euler_xyz(q)  # [X, Y, Z]
    order = order.upper()
    
    mapping = {
        'XYZ': [0,1,2],
        'XZY': [0,2,1],
        'YXZ': [1,0,2],
        'YZX': [1,2,0],
        'ZXY': [2,0,1],
        'ZYX': [2,1,0],
    }
    
    idx = mapping.get(order, mapping['ZYX'])
    return [base[i] for i in idx]


def mat4_to_bvh_values(mat4, is_hips=False, output_order='ZYX', negate=True):
    # Position: mat4의 translation 부분
    position = [mat4[3][0], mat4[3][1], mat4[3][2]]  # x, y, z
    
    # mat4에서 quaternion 추출
    quat = glm.quat_cast(mat4)
    
    # 정확한 euler angle 변환
    euler_radians = quat_to_euler_by_order(quat, output_order)
    rotation = [math.degrees(angle) for angle in euler_radians]
    
    # 부호 뒤집기
    if negate:
        rotation = [-r for r in rotation]
    
    if is_hips:
        return position + rotation  # [x,y,z, r1,r2,r3]
    return rotation  # [r1,r2,r3]

def extract_joint_kinematics(root, joint_name):
    """특정 joint의 kinematics (glm::mat4) 추출. Hips는 virtual_root * hips"""
    if joint_name == 'Hips':
        # virtual_root 찾기 (root가 virtual일 수 있음)
        virtual_root = root if 'virtual' in getattr(root, 'name', '').lower() else find_joint_by_name(root, 'virtual_root') or root
        hips = find_joint_by_name(virtual_root, 'Hips')
        
        if not hips:
            print(f"Warning: Hips not found!")
            return glm.mat4(1.0)  # Identity
        
        virtual_kin = getattr(virtual_root, 'kinematics', glm.mat4(1.0))
        hips_kin = getattr(hips, 'kinematics', glm.mat4(1.0))
        
        # GLM 곱셈
        final_kin = virtual_kin * hips_kin
        return final_kin
    
    joint = find_joint_by_name(root, joint_name)
    if not joint:
        print(f"Warning: Joint '{joint_name}' not found!")
        return glm.mat4(1.0)
    
    return getattr(joint, 'kinematics', glm.mat4(1.0))

def extract_current_frame_data(root, joint_rotation_orders=None):
    """현재 프레임의 모든 joint 데이터를 BVH 순서대로 추출 (각 joint의 rotation order 적용)"""
    joint_data = []
    
    for joint_name in BVH_JOINT_ORDER:
        kinematics = extract_joint_kinematics(root, joint_name)  # glm::mat4
        
        # 해당 joint의 rotation order 가져오기 (없으면 기본값 ZYX)
        if joint_rotation_orders and joint_name in joint_rotation_orders:
            rotation_order = joint_rotation_orders[joint_name]
        else:
            rotation_order = 'ZYX'  # BVH 기본값
        
        bvh_values = mat4_to_bvh_values(
            kinematics, 
            is_hips=(joint_name == 'Hips'),
            output_order=rotation_order
        )
        
        joint_data.extend(bvh_values)
    
    return np.array(joint_data, dtype=float).flatten()

def read_bvh_header(template_path):
    header_lines = []
    with open(template_path, 'r') as f:
        in_hierarchy = False
        for line in f:
            line = line.rstrip()
            if line.startswith('HIERARCHY'):
                in_hierarchy = True
            if line.startswith('MOTION'):
                break
            if in_hierarchy:
                header_lines.append(line)
    return '\n'.join(header_lines) + '\n'

def write_bvh_from_frames(root, all_frames_data, output_path, template_bvh_path, frame_time=0.016667):
    num_frames = len(all_frames_data)
    print(f"Writing BVH with {num_frames} frames to {output_path}")
    
    # BVH 파일에서 rotation order 파싱
    joint_rotation_orders = parse_bvh_rotation_orders(template_bvh_path)
    
    bvh_header = read_bvh_header(template_bvh_path)
    
    motion_lines = []
    for frame_idx, frame_data in enumerate(all_frames_data):
        # 프레임 데이터 적용
        populate_kinematics_dfs(root, frame_data)
        
        # rotation order를 적용해서 추출
        frame_bvh = extract_current_frame_data(root, joint_rotation_orders)
        
        # 문자열로 변환
        frame_str = ' '.join(f"{val:.6f}" for val in frame_bvh)
        motion_lines.append(frame_str)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{num_frames}")
    
    with open(output_path, 'w') as f:
        f.write(bvh_header)
        f.write("MOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        f.write('\n'.join(motion_lines) + '\n')
    
    print(f"BVH saved to {output_path}")
    return output_path

def frames_to_bvh(root, all_frames_data, output_filename="debug_output.bvh", template_bvh="data/template.bvh"):
    try:
        return write_bvh_from_frames(root, all_frames_data, output_filename, template_bvh)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None