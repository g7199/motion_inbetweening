import argparse
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio
from pyglm import glm
import sys
import math
import torch

# --- 사용자 정의 모듈 임포트 (기존 코드 가정) ---
from bvh_tools.Rendering import draw_humanoid, draw_trajectory
from bvh_tools.utils import draw_axes, set_lights
from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list  # BVH 파싱 모듈
from bvh_tools.virtual_transforms import extract_xz_plane  # virtual root axis용
from bvh_tools.Rendering import draw_virtual_root_axis  # 가정 (뷰어에 있음)
from bvh_tools.utils import random_color  # 색상 랜덤

# ----------------- 설정 -----------------
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
# ----------------------------------------

def create_fbo(width, height):
    """ FBO와 텍스처, 뎁스 버퍼를 생성하고 ID를 반환합니다. """
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("FBO 생성 실패!", file=sys.stderr)
        return None, None, None

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, texture, rbo

def save_video(frames, filename, fps):
    """ NumPy 배열로 된 프레임 목록을 동영상 파일로 저장합니다. """
    print(f"\n동영상을 저장하는 중입니다... 총 {len(frames)} 프레임")
    with imageio.get_writer(filename, fps=fps, quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"🎥 동영상이 {filename} 으로 성공적으로 저장되었습니다!")

def encode(bvh_paths, output_filename="rendering/output.mp4", trajectory=None):
    """
    BVH 파일(들)을 읽어서 렌더링하고 MP4로 저장.
    - bvh_paths: list of str (BVH 파일 경로들).
    - trajectory: optional, draw_trajectory용 데이터 (e.g., 로드된 데이터).
    """
    motions = []
    max_frames = 0
    for file_path in bvh_paths:
        if not os.path.exists(file_path):
            print(f"Warning: BVH 파일 '{file_path}'이 존재하지 않습니다. 스킵합니다.")
            continue
        
        root, motion = parse_bvh(file_path)
        joint_order = get_preorder_joint_list(root)
        motion.build_quaternion_frames(joint_order)
        virtual_root = motion.apply_virtual(root)  # virtual root 적용
        
        entry = {
            'name': os.path.basename(file_path),
            'root': virtual_root,
            'motion': motion,
            'frame_len': motion.frames,
            'color': random_color()  # 뷰어처럼 랜덤 색상
        }
        motions.append(entry)
        
        if motion.frames > max_frames:
            max_frames = motion.frames  # 가장 긴 모션에 맞춤
    
    if not motions:
        print("Error: 유효한 BVH 파일이 없습니다.")
        return None
    
    # FPS 계산: 첫 번째 motion의 frame_time 사용 (BVH 표준)
    fps = 1.0 / motions[0]['motion'].frame_time if hasattr(motions[0]['motion'], 'frame_time') else 60.0
    print(f"렌더링 FPS: {fps:.2f}")

    # 2. Pygame 및 OpenGL 초기화 (화면 없는 모드)
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)

    # 3. FBO 생성
    fbo, texture, rbo = create_fbo(*size)
    if fbo is None:
        pygame.quit()
        return None

    # 4. 렌더링 환경 및 카메라 설정 (뷰어 state 참고)
    glEnable(GL_DEPTH_TEST)
    set_lights()
    camera_eye = glm.vec3(60, 180, 600)  # 뷰어 초기값
    camera_center = glm.vec3(0, 80, 0)
    camera_up = glm.vec3(0, 1, 0)

    # 5. 프레임별 렌더링 루프
    recorded_frames = []
    total_frames = max_frames  # 총 프레임: 가장 긴 모션 기준

    for frame_idx in range(total_frames):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        glViewport(0, 0, *size)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, size[0] / size[1], 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(camera_eye.x, camera_eye.y, camera_eye.z,
                  camera_center.x, camera_center.y, camera_center.z,
                  camera_up.x, camera_up.y, camera_up.z)
        draw_axes()
        
        if trajectory is not None:
            draw_trajectory(trajectory)  # optional
        
        # 각 모션 렌더링 (뷰어처럼 motions 루프)
        for motion_entry in motions:
            # 프레임 적용 (루프: frame_idx % frame_len)
            local_idx = frame_idx % motion_entry['frame_len']
            motion_entry['motion'].apply_to_skeleton(local_idx, motion_entry['root'])
            
            # 휴머노이드 그리기
            
            draw_humanoid(motion_entry['root'], motion_entry['color'])
            
            # virtual root axis 그리기 (뷰어 느낌 반영)
            if motion_entry['root'].children:
                combined_kin = motion_entry['root'].kinematics * motion_entry['root'].children[0].kinematics
                draw_virtual_root_axis(combined_kin, motion_entry['color'])

        # 프레임 캡처
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        pixels = glReadPixels(0, 0, *size, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = np.flipud(image)
        recorded_frames.append(image)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        print(f"\r렌더링 진행률: {frame_idx + 1} / {total_frames}", end="")

    # 6. 리소스 정리 및 동영상 저장
    glDeleteRenderbuffers(1, [rbo])
    glDeleteTextures(1, [texture])
    glDeleteFramebuffers(1, [fbo])
    
    save_video(recorded_frames, output_filename, fps)
    pygame.quit()
    
    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVH 파일을 렌더링하여 MP4 비디오로 저장합니다.")
    parser.add_argument('--path', nargs='+', required=True, help="BVH 파일 경로(들). 여러 개 지정 가능.")
    parser.add_argument('--output', default="rendering/output.mp4", help="출력 MP4 파일명 (default: rendering/output.mp4)")
    parser.add_argument('--trajectory', default=None, help="Trajectory 데이터 파일 경로 (optional)")

    args = parser.parse_args()

    # trajectory 로드 (필요 시 구현. 여기서는 단순 str로 전달)
    trajectory_data = None
    if args.trajectory:
        # 예: trajectory 파일 로드 (실제 구현 필요, e.g., np.load(args.trajectory))
        print(f"Trajectory 로딩: {args.trajectory}")
        trajectory_data = torch.load(args.trajectory)  # placeholder: 실제 데이터 로드 코드 추가
        

    encode(args.path, args.output, trajectory=trajectory_data)