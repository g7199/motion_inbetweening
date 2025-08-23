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

# --- 사용자 정의 모듈 임포트 ---
from bvh_tools.Rendering import draw_humanoid, draw_trajectory, draw_positions_points_frame
from bvh_tools.utils import draw_axes, set_lights
from bvh_tools.reverse import populate_kinematics_dfs

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


# ==============================================================================
#  ↓↓↓ 요청하신 encode() 함수 ↓↓↓
# ==============================================================================
def encode(root, all_frames_data,
           trajectory=None, traj_mean=None, traj_std=None,
           positions=None,  pos_mean=None, pos_std=None,
           output_filename="rendering/output.mp4"):
    
    # 2. Pygame 및 OpenGL 초기화 (화면 없는 모드)
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)

    # 3. FBO(Frame Buffer Object) 생성
    fbo, texture, rbo = create_fbo(*size)
    if fbo is None:
        pygame.quit()
        return None

    # 4. 렌더링 환경 및 카메라 설정
    glEnable(GL_DEPTH_TEST)
    set_lights()
    camera_eye = glm.vec3(60, 180, 600)
    camera_center = glm.vec3(0, 80, 0)
    camera_up = glm.vec3(0, 1, 0)

    # 5. 프레임별 렌더링 루프
    recorded_frames = []
    total_frames = len(all_frames_data)
    fps = 1/0.016667

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
            # sample_motion_while_training에서 호출할 때
            draw_trajectory(trajectory[:, :2], traj_mean[:2], traj_std[:2])

        if positions is not None:
            cur_pos_frame = positions[frame_idx]   # [J, 3]
            draw_positions_points_frame(
                positions_frame=cur_pos_frame,
                pos_mean=pos_mean, pos_std=pos_std,
                point_size=3.0, color=(0.0, 1.0, 0.0),
                use_sphere=False, sphere_radius=2.5
            )
            
        populate_kinematics_dfs(root, all_frames_data[frame_idx])
        
        draw_humanoid(root, [0.2, 0.6, 0.9])

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