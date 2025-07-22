import argparse
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio
from pyglm import glm
import sys

# --- 사용자 정의 모듈 임포트 ---
from bvh_tools.bvh_controller import parse_bvh, get_preorder_joint_list
from bvh_tools.Rendering import draw_humanoid
from bvh_tools.utils import draw_axes, set_lights

# ----------------- 설정 -----------------
OUTPUT_FILENAME = "output/output_headless.mp4"
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
# ----------------------------------------

def create_fbo(width, height):
    """ FBO와 텍스처, 뎁스 버퍼를 생성하고 ID를 반환합니다. """
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # 렌더링 결과를 저장할 텍스처 생성
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # 깊이 테스트를 위한 렌더버퍼 생성
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

    # FBO가 완전한지 확인
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

def main(state):
    # Pygame 창을 화면에 표시하지 않고 초기화
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)

    # FBO 생성
    fbo, texture, rbo = create_fbo(*size)
    if fbo is None: return

    glEnable(GL_DEPTH_TEST)
    set_lights()

    # 카메라 설정
    camera_eye = glm.vec3(60, 180, 600)
    camera_center = glm.vec3(0, 80, 0)
    camera_up = glm.vec3(0, 1, 0)

    recorded_frames = []
    total_frames = state['frame_len']
    fps = round(1.0 / state['motion'].frame_time)

    for frame_idx in range(total_frames):
        # 1. 렌더링 대상을 FBO로 지정
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        
        # 2. 렌더링
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

        state['motion'].apply_to_skeleton(frame_idx, state['root'])
        draw_humanoid(state['root'], [0.2, 0.6, 0.9])

        # 3. FBO에서 픽셀 데이터 읽기
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        pixels = glReadPixels(0, 0, *size, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = np.flipud(image)
        recorded_frames.append(image)

        # 4. 렌더링 대상을 원래대로 복원
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        print(f"\r렌더링 진행률: {frame_idx + 1} / {total_frames}", end="")

    # 리소스 정리
    glDeleteRenderbuffers(1, [rbo])
    glDeleteTextures(1, [texture])
    glDeleteFramebuffers(1, [fbo])
    
    save_video(recorded_frames, OUTPUT_FILENAME, fps)
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a BVH file to video headlessly.")
    parser.add_argument("file_path", help="Path to the .bvh file.")
    args = parser.parse_args()

    root, motion = parse_bvh(args.file_path)
    joint_order = get_preorder_joint_list(root)
    motion.build_quaternion_frames(joint_order)
    virtual_root = motion.apply_virtual(root)
    
    state = {
        'root': virtual_root,
        'motion': motion,
        'frame_len': motion.frames,
    }

    main(state)