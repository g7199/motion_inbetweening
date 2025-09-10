import numpy as np
import math

def get_kinematics_dfs(joint, result_list, posis_list, relative_posis_list, contact_list=None,
                       prev_foot_positions=None, dt=1/60.0,
                       cur_pos=np.zeros(3), cur_rot=np.eye(3),
                       include_pos=True, is_root=True, height_threshold=5.0):
    
    # 4개 foot joint 구분 (기존 RightToe, LeftToe에서 확장)
    foot_joints = {
        "RightAnkle": 0,  # right_heel index
        "RightToe": 1,    # right_toe index
        "LeftAnkle": 2,   # left_heel index  
        "LeftToe": 3      # left_toe index
    }
    
    T = np.array(joint.kinematics)   # [4,4] 로컬
    R_local = T[:3, :3]
    p_local = T[:3, 3]

    # 부모 좌표계 → 현재 로컬 루트 기준
    p_global = cur_rot @ p_local + cur_pos
    R_global = cur_rot @ R_local

    if is_root:
        result_list.append(np.array([p_local[1]]))  # root height

    if joint.name != "End Site":
        result_list.append(R_local[:, :2].T.flatten())  # 6D rot
        if (include_pos and not is_root):
            posis_list.append(p_global)  # velocity 계산용
            relative_posis_list.append(p_global)

        if contact_list is not None and joint.name in foot_joints:
            height_contact = int(p_global[1] < height_threshold)  # binary 0 or 1
            foot_idx = foot_joints[joint.name]
            contact_list.append((foot_idx, height_contact, p_global.copy()))

    for child in joint.children:
        get_kinematics_dfs(child, result_list, posis_list, relative_posis_list, contact_list,
                          prev_foot_positions, dt, p_global, R_global, 
                          include_pos=include_pos, is_root=False)
        
def sample_data_with_endpoints(data, ratio):
   
   total_length = len(data)
   target_count = max(2, int(total_length * ratio))  # 최소 2개 (첫번째, 마지막)
   
   indices = np.linspace(0, total_length - 1, target_count, dtype=int)
   sampled_data = data[indices]
   
   return sampled_data, indices


def get_data(motion, virtual_root, time_size=180, start_frame=0, include_pos=True, 
             height_threshold=5.0, velocity_threshold=2.0):
    data = []
    prev_yaw = None
    prev_global_pos_xz = None
    global_pos_xz = []
    
    # 4차원으로 변경: [right_heel, right_toe, left_heel, left_toe] (int으로 binary)
    foot_contacts = np.zeros((time_size, 4), dtype=int)
    
    # 각 발 부위별 position 저장 (velocity 계산용)
    foot_positions = [
        np.zeros((time_size, 3)),  # right_heel (index 0)
        np.zeros((time_size, 3)),  # right_toe (index 1)
        np.zeros((time_size, 3)),  # left_heel (index 2)  
        np.zeros((time_size, 3))   # left_toe (index 3)
    ]
    
    # 모든 관절의 position 저장 (velocity 계산용)
    all_joint_positions = []
    # 상대 position 저장
    all_relative_positions = []
    
    global_rot = None
    local_rot = None
    
    for i in range(start_frame, start_frame + time_size):
        motion.apply_to_skeleton(i, virtual_root)
        root = virtual_root.children[0]

        if global_rot is None:
            global_rot = np.array(virtual_root.kinematics)[:3, :3]
        if local_rot is None:
            local_rot = np.array(root.kinematics)[:3, :3]

        # Root processing (기존과 동일)
        current_global_kin = np.array(virtual_root.kinematics)
        current_global_rot = current_global_kin[:3, :3]
        current_global_pos = current_global_kin[:3, 3]
        current_global_pos = local_rot @ global_rot.T @ current_global_pos
        current_yaw = math.atan2(current_global_rot[0, 2], current_global_rot[2, 2])

        if prev_yaw is None:
            angular_velocity = 0.0
        else:
            angular_velocity = current_yaw - prev_yaw
            if angular_velocity > math.pi:  angular_velocity -= 2 * math.pi
            if angular_velocity < -math.pi: angular_velocity += 2 * math.pi
        prev_yaw = current_yaw

        current_global_pos_xz = np.array([current_global_pos[0], current_global_pos[2]])
        if prev_global_pos_xz is None:
            linear_velocity = np.array([0.0, 0.0])
        else:
            linear_velocity = current_global_pos_xz - prev_global_pos_xz
        prev_global_pos_xz = current_global_pos_xz
        trajectory_data = np.array([linear_velocity[0], linear_velocity[1], angular_velocity])
        global_pos_xz.append(trajectory_data)

        # Extract kinematics
        parts = []
        posis = []
        relative_posis = []
        foot_frame = []
        
        get_kinematics_dfs(root, parts, posis, relative_posis, foot_frame, 
                          include_pos=include_pos, 
                          height_threshold=height_threshold)

        # posis를 저장 (나중에 velocity로 변환)
        all_joint_positions.append(posis)
        # 상대 position 저장
        all_relative_positions.append(relative_posis)

        # foot_frame에서 height_contact와 position 추출
        frame_contact = [0, 0, 0, 0]
        
        for foot_idx, height_contact, position in foot_frame:
            frame_contact[foot_idx] = height_contact
            foot_positions[foot_idx][i-start_frame] = position
        
        foot_contacts[i-start_frame] = frame_contact  # 일단 height_contact만 저장 (velocity는 나중에 hybrid)

        # Feature 구성 - rotation만 먼저 추가
        pose_features = np.concatenate(parts)
        
        # 임시로 저장 (velocity 계산 후 다시 구성)
        data.append(pose_features)

    # Position을 Velocity로 변환 (dt 없이 단순 차이)
    joint_velocities = []
    
    for t in range(time_size):
        if t == 0:
            # 첫 번째 프레임은 0으로 초기화
            frame_velocities = [np.zeros(3) for _ in all_joint_positions[0]]
        else:
            # 이전 프레임과의 차이로 velocity 계산
            frame_velocities = []
            for j in range(len(all_joint_positions[t])):
                vel = all_joint_positions[t][j] - all_joint_positions[t-1][j]
                frame_velocities.append(vel)
        
        # flatten해서 저장
        joint_velocities.append(np.concatenate(frame_velocities))

    # Velocity calculation for all 4 foot parts (contact detection용 - dt 사용)
    dt = 1/60.0
    foot_velocities = []
    
    for foot_pos in foot_positions:
        vel = np.linalg.norm(np.diff(foot_pos, axis=0), axis=1) / dt  # 물리적 속도
        vel = np.insert(vel, 0, 0.0)  # Add 0 for first frame
        foot_velocities.append(vel)

    # Hybrid contact detection (height AND velocity) for all 4 parts (binary 0 or 1)
    for i in range(4):
        vel = foot_velocities[i]
        vel_contact = (vel < velocity_threshold).astype(int)
        foot_contacts[:, i] = foot_contacts[:, i] & vel_contact  # binary AND

    # 최종 데이터 구성: [trajectory, pose, relative_positions, joint_velocities, foot_contacts]
    data_with_features = []
    for t in range(time_size):
        # 상대 position을 flatten
        relative_pos_flattened = np.concatenate(all_relative_positions[t])
        
        frame_data = np.concatenate([
            global_pos_xz[t],           # [3] - linear_vel_x, linear_vel_z, angular_vel
            data[t],                    # pose features (rotations)
            relative_pos_flattened,     # relative positions (이미 root 기준)
            #joint_velocities[t],        # joint velocities (positions → velocities)
            foot_contacts[t]            # [4] - foot contact binary (0 or 1)
        ])
        data_with_features.append(frame_data)
    
    data_with_features = np.array(data_with_features)  # [T, feature_dim + relative_pos_dim + 4]
    
    return data_with_features, np.array(global_pos_xz)