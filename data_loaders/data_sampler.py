import numpy as np

def get_kinematics_dfs(joint, result_list, is_root=True):
    rot = np.array(joint.kinematics)
    if is_root:
        result_list.append(rot[:3, 3].flatten())
    result_list.append(rot[:2, :3].flatten())

    for child in joint.children:
        get_kinematics_dfs(child, result_list, is_root=False)

def get_data(motion, root, time_size=180, start_frame=0):
    data = []
    for i in range(start_frame, start_frame+time_size, 1):
        motion.apply_to_skeleton(i, root)
        all_parts = []
        get_kinematics_dfs(root, all_parts)
        data.append(np.concatenate(all_parts))

    return np.array(data)

def get_statistics(bvh_files, clip_length=180):
    from tqdm import tqdm
    from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
    
    feature_dim = 171 # 특징 벡터의 차원
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
                clip_data = get_data(motion, root, clip_length, start_frame)

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
        