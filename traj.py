import argparse
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from pyglm import glm  # glm 라이브러리 필요 (pip install pyglm)
import sys
import math
import torch
from datetime import datetime

WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720

class CylinderController:
    def __init__(self, max_speed=50.0):
        self.position = glm.vec3(0.0, 10.0, 0.0)  # 초기 위치 (y=10으로 바닥 위에)
        self.current_forward = glm.vec3(0, 0, 1)  # 초기 방향 (+Z)
        self.max_speed = max_speed
        self.radius = 5.0  # 원통 반지름
        self.height = 20.0  # 원통 높이
        
        # 물리 파라미터
        self.acceleration = 1.2
        self.deceleration = 3.0
        self.current_velocity = glm.vec3(0)
        
        # 궤적 기록
        self.trajectory = []  # [{'x': float, 'z': float, 'yaw': float, 'timestamp': float}, ...]
        self.recording = False
        
        # 키 상태
        self.keys = {
            'w': False, 's': False, 'a': False, 'd': False
        }

    def update_keys(self, keys_pressed):
        """현재 키 상태 업데이트"""
        self.keys['w'] = keys_pressed[pygame.K_w]
        self.keys['s'] = keys_pressed[pygame.K_s]
        self.keys['a'] = keys_pressed[pygame.K_a]
        self.keys['d'] = keys_pressed[pygame.K_d]

    def compute_velocity(self, delta_time):
        """입력에 따른 목표 속도 계산 (사용자 요구: W=+Z 앞으로, D=-X 오른쪽)"""
        vel = glm.vec3(0)
        if self.keys['w']: vel += glm.vec3(0, 0, 1)   # +Z 앞으로
        if self.keys['s']: vel += glm.vec3(0, 0, -1)  # -Z 뒤로
        if self.keys['a']: vel += glm.vec3(1, 0, 0)   # +X 왼쪽으로
        if self.keys['d']: vel += glm.vec3(-1, 0, 0)  # -X 오른쪽으로

        if glm.length(vel) > 0:
            vel = glm.normalize(vel) * self.max_speed
            # 가속
            self.current_velocity += (vel - self.current_velocity) * self.acceleration * delta_time
        else:
            # 감속
            self.current_velocity -= self.current_velocity * self.deceleration * delta_time
            
        return self.current_velocity

    def update_direction(self, velocity, delta_time):
        """속도 방향에 따라 원통의 방향 업데이트"""
        if glm.length(velocity) > 0.1:  # 임계값 이상으로 움직일 때만
            desired_direction = glm.normalize(velocity)
            
            # 현재 방향에서 목표 방향으로 부드럽게 회전
            prev_forward = glm.normalize(self.current_forward)
            
            # 쿼터니언을 사용한 부드러운 회전
            q_current = glm.quat(glm.vec3(0, 0, 1), prev_forward)
            q_target = glm.quat(glm.vec3(0, 0, 1), desired_direction)
            
            # 속도에 따른 회전 속도 조절
            speed = glm.length(velocity)
            slerp_amount = min(0.2, speed * 0.02)
            
            q_new = glm.slerp(q_current, q_target, slerp_amount)
            self.current_forward = glm.normalize(q_new * glm.vec3(0, 0, 1))

    def get_yaw_angle(self):
        """현재 방향에서 yaw 각도 계산 (라디안)"""
        return math.atan2(self.current_forward.x, self.current_forward.z)

    def update(self, dt):
        """물체 위치 및 회전 업데이트"""
        # 속도 계산
        velocity = self.compute_velocity(dt)
        
        # 방향 업데이트
        self.update_direction(velocity, dt)
        
        # 위치 업데이트
        self.position += velocity * dt
        
        # 궤적 기록 (움직임이 있을 때만)
        if self.recording and glm.length(velocity) > 0.1:
            timestamp = pygame.time.get_ticks() / 1000.0
            self.trajectory.append({
                'x': float(self.position.x),
                'z': float(self.position.z),
                'yaw': float(self.get_yaw_angle()),
                'timestamp': timestamp
            })

    def draw(self):
        """원통 그리기"""
        glPushMatrix()
        glTranslatef(self.position.x, self.position.y, self.position.z)
        
        # 방향에 따른 회전
        yaw = self.get_yaw_angle()
        glRotatef(math.degrees(yaw), 0, 1, 0)
        
        # 원통 색상 (밝은 파란색)
        glColor3f(0.3, 0.6, 1.0)
        
        # 원통 그리기 (GLU 사용)
        quadric = gluNewQuadric()
        gluQuadricDrawStyle(quadric, GLU_FILL)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        
        # 원통 몸체
        gluCylinder(quadric, self.radius, self.radius, self.height, 32, 1)
        
        # 윗면
        glPushMatrix()
        glTranslatef(0, 0, self.height)
        gluDisk(quadric, 0, self.radius, 32, 1)
        glPopMatrix()
        
        # 아랫면
        glPushMatrix()
        glRotatef(180, 1, 0, 0)
        gluDisk(quadric, 0, self.radius, 32, 1)
        glPopMatrix()
        
        # 방향 표시 (앞쪽에 작은 원뿔)
        glColor3f(1.0, 0.2, 0.2)  # 빨간색
        glPushMatrix()
        glTranslatef(0, 0, self.height + 2)
        gluCylinder(quadric, 2, 0, 8, 16, 1)
        glPopMatrix()
        
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def draw_trajectory_line(self):
        """기록된 궤적을 선으로 그리기"""
        if len(self.trajectory) < 2:
            return
            
        glColor3f(1.0, 1.0, 0.0)  # 노란색
        glLineWidth(3.0)
        glBegin(GL_LINE_STRIP)
        for point in self.trajectory:
            glVertex3f(point['x'], 5.0, point['z'])  # y=5 높이에서 궤적 표시
        glEnd()
        glLineWidth(1.0)

    def start_recording(self):
        """궤적 기록 시작"""
        self.recording = True
        self.trajectory.clear()
        print("🔴 궤적 기록 시작")

    def stop_recording(self):
        """궤적 기록 중지"""
        self.recording = False
        print(f"🟡 궤적 기록 중지 ({len(self.trajectory)} 포인트)")

    def resample_to_180(self):
        """궤적을 180개 포인트로 강제 리샘플링 (선형 보간) 후 속도로 변환 (dt=1 고정, 첫 번째 0, yaw degree)"""
        if not self.trajectory:
            return None
        
        N = len(self.trajectory)
        if N == 0:
            return None
        
        times = np.array([p['timestamp'] for p in self.trajectory])
        xs = np.array([p['x'] for p in self.trajectory])
        zs = np.array([p['z'] for p in self.trajectory])
        yaws = np.array([p['yaw'] for p in self.trajectory])
        
        total_time = times[-1] - times[0] if times[-1] != times[0] else 1e-6  # 0 방지
        
        # 위치 리샘플링
        if N == 1 or total_time <= 0:
            # 포인트 부족: 모든 속도 0
            return torch.zeros(180, 3, dtype=torch.float32)
        else:
            t_norm = (times - times[0]) / total_time
            target_t = np.linspace(0, 1, 180)
            target_xs = np.interp(target_t, t_norm, xs)
            target_zs = np.interp(target_t, t_norm, zs)
            target_yaws = np.interp(target_t, t_norm, yaws)  # yaw를 degree로 변환
        
        resampled = np.column_stack((target_xs, target_zs, target_yaws))
        
        # 속도 계산 (dt=1 고정, 나누기 없음, 첫 번째 0)
        diff = np.diff(resampled, axis=0)  # (179, 3) 차이
        velocity = np.zeros((180, 3))
        velocity[1:] = diff  # velocity[0] = [0,0,0], velocity[1:180] = diff
        
        return torch.tensor(velocity, dtype=torch.float32)

    def save_trajectory(self, filename=None):
        """궤적을 (180, 3) velocity tensor로 .pt 파일에 저장 (dt=1 fixed, yaw in degrees)"""
        tensor = self.resample_to_180()
        if tensor is None:
            print("저장할 궤적이 없습니다.")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.pt"
            
        torch.save(tensor, filename)
        print(f"📁 속도 궤적이 {filename}에 저장되었습니다. (shape: {tensor.shape}, dtype: float32, [vx, vz, vyaw(deg)], dt=1 fixed, first=0)")
        return filename

def draw_grid(size=500, step=50):
    """바닥에 그리드 그리기"""
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)
    for i in range(-size, size + 1, step):
        # X 방향 선
        glVertex3f(-size, 0, i)
        glVertex3f(size, 0, i)
        # Z 방향 선
        glVertex3f(i, 0, -size)
        glVertex3f(i, 0, size)
    glEnd()

def draw_axes():
    """좌표축 그리기"""
    glLineWidth(2.0)
    glBegin(GL_LINES)
    # X축 (빨강)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(100, 0, 0)
    # Y축 (녹색)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 100, 0)
    # Z축 (파랑)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 100)
    glEnd()
    glLineWidth(1.0)

def interactive_mode():
    """
    대화형 모드 - 원통을 조종하면서 궤적 기록 및 저장
    """
    # Pygame 및 OpenGL 초기화
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    screen = pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Cylinder Trajectory Recorder")

    # OpenGL 설정
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 100, 0, 0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])

    # 카메라 및 컨트롤러 초기화
    cylinder = CylinderController()
    camera_distance = 100.0
    camera_height = 50.0

    clock = pygame.time.Clock()
    running = True

    print("=== 조작법 ===")
    print("WASD: 원통 이동 (방향 자동 조절)")
    print("R: 궤적 기록 시작/중지")
    print("T: 궤적 저장 (.pt 파일, (180,3) velocity tensor)")
    print("SPACE: 종료")
    print("===============")

    while running:
        dt = clock.tick(60) / 1000.0  # 델타 타임 (초)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False
                elif event.key == pygame.K_r:
                    if cylinder.recording:
                        cylinder.stop_recording()
                    else:
                        cylinder.start_recording()
                elif event.key == pygame.K_t:
                    cylinder.save_trajectory()

        # 키 상태 업데이트 및 물체 이동
        keys_pressed = pygame.key.get_pressed()
        cylinder.update_keys(keys_pressed)
        cylinder.update(dt)

        # 카메라 위치 (원통 뒤에서 따라가는 3인칭 시점)
        yaw = cylinder.get_yaw_angle()
        camera_offset_x = -math.sin(yaw) * camera_distance
        camera_offset_z = -math.cos(yaw) * camera_distance
        camera_eye = glm.vec3(
            cylinder.position.x + camera_offset_x,
            cylinder.position.y + camera_height,
            cylinder.position.z + camera_offset_z
        )
        camera_center = glm.vec3(cylinder.position.x, cylinder.position.y, cylinder.position.z)
        camera_up = glm.vec3(0, 1, 0)

        # 렌더링
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, size[0] / size[1], 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(camera_eye.x, camera_eye.y, camera_eye.z,
                  camera_center.x, camera_center.y, camera_center.z,
                  camera_up.x, camera_up.y, camera_up.z)

        # 씬 그리기
        draw_grid()
        draw_axes()
        
        # 궤적 그리기
        cylinder.draw_trajectory_line()
        
        # 원통 그리기
        cylinder.draw()

        pygame.display.flip()

    # 종료 시 궤적 자동 저장
    if cylinder.trajectory:
        cylinder.save_trajectory()

    pygame.quit()

if __name__ == "__main__":
    interactive_mode()