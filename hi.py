import numpy as np
from data_loaders.data_loader import get_data
from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list


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
                clip_data, _ = get_data(motion, virtual_root, clip_length, start_frame)

                if not np.isfinite(clip_data).all():
                    nan_count = np.isnan(clip_data).sum()
                    inf_count = np.isinf(clip_data).sum()
                    print(f"[경고] {file_path} start_frame={start_frame} 에서 clip_data에 "
                        f"{nan_count} NaN, {inf_count} Inf 발견")
                
                
                # 자신만의 통계치를 누적
                local_count += clip_data.shape[0] 
                local_sum += np.sum(clip_data, axis=0)
                local_sum_sq += np.sum(clip_data**2, axis=0)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    print(f"Finished processing file: {file_path}")
    return local_count, local_sum, local_sum_sq

def get_statistics_parallel(bvh_dir, clip_length=180, feature_dim=212):
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
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)

    print("\n계산 완료!")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    # 파일 저장
    np.save('data/mean_pos.npy', mean)
    np.save('data/std_pos.npy', std)
        
if __name__ == '__main__':
    import glob

    bvh_dir = "bvh_set"

    get_statistics_parallel(bvh_dir)