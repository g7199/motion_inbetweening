import numpy as np
import math

def get_kinematics_dfs(joint, result_list, posis_list,
                       cur_pos=np.zeros(3), cur_rot=np.eye(3),
                       include_pos=True, is_root=True):
    T = np.array(joint.kinematics)   # [4,4] 로컬
    R_local = T[:3, :3]
    p_local = T[:3, 3]

    # 부모 좌표계 → 현재 글로벌
    p_global = cur_rot @ p_local + cur_pos
    R_global = cur_rot @ R_local

    if is_root:
        result_list.append(np.array([p_local[1]]))

    if joint.name != "End Site":
        result_list.append(R_local[:, :2].T.flatten())  # 6D rot
        if include_pos:
            posis_list.append(p_global)

    for child in joint.children:
        get_kinematics_dfs(child, result_list, posis_list,
                           p_global, R_global, include_pos=include_pos, is_root=False)



def get_data(motion, virtual_root, time_size=180, start_frame=0, include_pos=True):
    data = []
    prev_yaw = None
    prev_global_pos_xz = None

    for i in range(start_frame, start_frame + time_size):
        motion.apply_to_skeleton(i, virtual_root)
        root = virtual_root.children[0]

        current_global_kin = np.array(virtual_root.kinematics)
        current_global_rot = current_global_kin[:3, :3]
        current_global_pos = current_global_kin[:3, 3]
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

        parts = []
        posis = []
        get_kinematics_dfs(root, parts, posis, include_pos=include_pos)  # 여기만 True로
        pose_features = np.concatenate(parts)
        posis_features = np.concatenate(posis)

        final_feature = np.concatenate([linear_velocity, np.array([angular_velocity]), pose_features, posis_features])
        data.append(final_feature)

    return np.array(data)


def get_statistics(bvh_dir, clip_length=180, feature_dim=172):
    from tqdm import tqdm
    from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
    import glob

    bvh_files = sorted(glob.glob(f"{bvh_dir}/**/*.bvh", recursive=True))
    print(f"Found {len(bvh_files)} BVH files in {bvh_dir}")
    
    count = 0
    mean = np.zeros(feature_dim)
    M2 = np.zeros(feature_dim) # 제곱합의 차이를 저장


    for file_path in bvh_files:
        root, motion = parse_bvh(file_path)
        joint_order = get_preorder_joint_list(root)
        motion.build_quaternion_frames(joint_order)
        virtual_root = motion.apply_virtual(root)
        
        if motion.frames >= clip_length:
            for start_frame in tqdm(range(motion.frames - clip_length + 1)):
                clip_data = get_data(motion, virtual_root, clip_length, start_frame)

                for frame_vector in clip_data:
                    count += 1
                    delta = frame_vector - mean
                    mean += delta / count
                    delta2 = frame_vector - mean
                    M2 += delta * delta2

    if count > 0:
        std = np.sqrt(M2 / count)
        print("\nDone!")
        print(f"Mean shape: {mean.shape}")
        print(f"Std shape: {std.shape}")
    else:
        print("No calculated data.")

    np.save('data/mean.npy', mean)
    np.save('data/std.npy', std)

def process_file_for_stats(file_path, clip_length, feature_dim):
    from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
    print(f"Processing file: {file_path}")
    
    local_count = 0
    local_sum = np.zeros(feature_dim)
    local_sum_sq = np.zeros(feature_dim) # 제곱의 합

    try:
        root, motion = parse_bvh(file_path)
        joint_order = get_preorder_joint_list(root)
        motion.build_quaternion_frames(joint_order)
        virtual_root = motion.apply_virtual(root)
        
        if motion.frames >= clip_length:
            for start_frame in range(motion.frames - clip_length + 1):
                clip_data = get_data(motion, virtual_root, clip_length, start_frame)
                
                # 자신만의 통계치를 누적
                local_count += clip_data.shape[0] 
                local_sum += np.sum(clip_data, axis=0)
                local_sum_sq += np.sum(clip_data**2, axis=0)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    print(f"Finished processing file: {file_path}")
    return local_count, local_sum, local_sum_sq

def get_statistics_parallel(bvh_dir, clip_length=180, feature_dim=172):
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import glob

    bvh_files = sorted(glob.glob(f"{bvh_dir}/**/*.bvh", recursive=True))
    print("병렬 처리를 시작합니다...")
    
    with Parallel(n_jobs=-1) as parallel:
        
        # 1. 실행할 작업 목록을 미리 준비합니다.
        tasks = (delayed(process_file_for_stats)(fp, clip_length, feature_dim) for fp in bvh_files)
        
        # 2. parallel()이 반환하는 결과 이터레이터(iterator)를 tqdm으로 감쌉니다.
        results = tqdm(parallel(tasks), total=len(bvh_files))
        
        # 3. 결과를 리스트로 변환합니다. 이 과정에서 프로그레스 바가 표시됩니다.
        results_list = list(results)
    
    # 각 워커가 반환한 결과들을 취합
    total_count = 0
    total_sum = np.zeros(feature_dim)
    total_sum_sq = np.zeros(feature_dim)
    
    for local_count, local_sum, local_sum_sq in results:
        total_count += local_count
        total_sum += local_sum
        total_sum_sq += local_sum_sq
        
    if total_count == 0:
        print("처리된 데이터가 없습니다.")
        return

    # 최종 평균과 표준편차 계산
    mean = total_sum / total_count
    # 분산 = (제곱의 평균) - (평균의 제곱)
    variance = (total_sum_sq / total_count) - (mean ** 2)
    std = np.sqrt(variance)

    print("\n계산 완료!")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    # 파일 저장
    np.save('data/mean.npy', mean)
    np.save('data/std.npy', std)
        