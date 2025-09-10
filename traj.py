import pygame
from pygame import Rect
import numpy as np
from typing import Optional
import torch
import math
import json
from datetime import datetime

def to_unit(traj: np.ndarray, pos_scale: float) -> np.ndarray:
    out = traj.copy()
    out[:, :2] *= pos_scale  # 위치만 스케일, 각도(rad)는 그대로
    return out

class TrajectoryDrawer:
    def __init__(
        self,
        width=800,
        height=600,
        clip_length=180,       # 최종 샘플 수
        record_hz=60,          # 녹화 주파수(Hz)
        duration_seconds=None, # 주어지면 clip_length = int(record_hz * duration_seconds)
        world_scale=4.0,       # 화면 표시용 스케일
        walk_speed=50.0,       # 기본 속도 (3D와 유사하게)
        run_speed=100.0,       # 빠른 속도
        crouch_speed=25.0      # 느린 속도
    ):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Trajectory Drawer (3D Compatible)")

        # 길이 결정
        self.record_hz = record_hz
        if duration_seconds is not None:
            self.clip_length = int(record_hz * duration_seconds)
        else:
            self.clip_length = clip_length

        # 색상
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (64, 64, 64)
        self.LIGHT_GRAY = (128, 128, 128)

        # 격자/월드
        self.grid_size = 20
        self.center_x = width // 2
        self.center_y = height // 2
        self.world_scale = world_scale

        # 상태 (OpenGL 좌표계: X=오른쪽, Z=앞쪽)
        self.position = np.array([0.0, 0.0], dtype=float)  # [X, Z]
        self.theta = 0.0  # yaw 각도 (라디안)
        
        # 속도 파라미터
        self.walk_speed = walk_speed
        self.run_speed = run_speed
        self.crouch_speed = crouch_speed

        # 녹화 관련
        self.recording = False
        self.trajectory = []  # 3D 호환 형식
        self.traj_buf = np.zeros((self.clip_length, 3), dtype=np.float32)
        self.sample_idx = 0
        self.sample_accum = 0.0

        # UI
        self.start_button = Rect(10, 10, 100, 30)
        self.clear_button = Rect(120, 10, 80, 30)
        self.done_button = Rect(210, 10, 80, 30)
        self.font = pygame.font.Font(None, 24)

    # 좌표 변환 (OpenGL 좌표계: X=오른쪽, Z=앞쪽=화면위쪽)
    def world_to_pixel(self, world_pos):
        px = world_pos[0] * self.world_scale + self.center_x
        py = -world_pos[1] * self.world_scale + self.center_y  # Z축 뒤집기 (앞쪽이 화면 위쪽)
        return (int(px), int(py))

    def pixel_to_world(self, pixel_pos):
        wx = (pixel_pos[0] - self.center_x) / self.world_scale
        wz = -(pixel_pos[1] - self.center_y) / self.world_scale  # Z축 뒤집기
        return np.array([wx, wz], dtype=float)

    # 마우스 회전 (간단하게)
    def update_theta_look_at_mouse(self):
        mx, my = pygame.mouse.get_pos()
        m_w = self.pixel_to_world((mx, my))
        d = m_w - self.position  # [dX, dZ]
        # 3D 호환: atan2(x, z)
        self.theta = math.atan2(d[0], d[1])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.recording and self.sample_idx == 0:
                    self.start_recording()
                elif event.key == pygame.K_r:
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif event.key == pygame.K_t:
                    self.save_trajectory()
                elif event.key == pygame.K_c:
                    self.clear_trajectory()
                elif event.key == pygame.K_ESCAPE:
                    return False
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.start_button.collidepoint(mouse_pos) and not self.recording:
                    self.start_recording()
                elif self.clear_button.collidepoint(mouse_pos):
                    self.clear_trajectory()
                elif self.done_button.collidepoint(mouse_pos):
                    if self.sample_idx == self.clip_length:
                        return False
        return True

    def update(self, dt):
        # WASD + 마우스 컨트롤 (캐릭터 방향 기준)
        keys = pygame.key.get_pressed()
        
        # 로컬 이동 방향 입력 (캐릭터 기준)
        local_move = np.array([0.0, 0.0], dtype=float)  # [forward/back, left/right]

        # WASD (캐릭터 방향 기준)
        if keys[pygame.K_w]: local_move[0] += 1.0   # 앞으로
        if keys[pygame.K_s]: local_move[0] -= 1.0   # 뒤로
        if keys[pygame.K_a]: local_move[1] -= 1.0   # 왼쪽
        if keys[pygame.K_d]: local_move[1] += 1.0   # 오른쪽
        
        # 방향키 (절대 방향)
        absolute_move = np.array([0.0, 0.0], dtype=float)  # [X, Z]
        if keys[pygame.K_UP]:    absolute_move[1] += 1.0   # +Z (화면에서 위쪽)
        if keys[pygame.K_DOWN]:  absolute_move[1] -= 1.0   # -Z (화면에서 아래쪽)
        if keys[pygame.K_LEFT]:  absolute_move[0] -= 1.0   # -X (화면에서 왼쪽)
        if keys[pygame.K_RIGHT]: absolute_move[0] += 1.0   # +X (화면에서 오른쪽)

        # 로컬 이동을 월드 좌표로 변환 (캐릭터 회전 적용)
        world_move = np.array([0.0, 0.0], dtype=float)
        if np.linalg.norm(local_move) > 0:
            # 캐릭터의 forward 방향 (theta 기준)
            forward = np.array([math.sin(self.theta), math.cos(self.theta)])  # [X, Z]
            right = np.array([math.cos(self.theta), -math.sin(self.theta)])   # [X, Z]
            
            world_move = local_move[0] * forward + local_move[1] * right

        # 절대 이동과 상대 이동 합치기
        total_move = world_move + absolute_move

        # 정규화
        norm = np.linalg.norm(total_move)
        if norm > 0:
            total_move /= norm

        # 속도 선택
        speed = self.walk_speed
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            speed = self.run_speed
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            speed = self.crouch_speed

        # 이동
        displacement = total_move * speed * dt
        self.position += displacement

        # 마우스 방향
        self.update_theta_look_at_mouse()

        # 3D 호환 궤적 기록
        if self.recording and norm > 0:
            timestamp = pygame.time.get_ticks() / 1000.0
            self.trajectory.append({
                'x': float(self.position[0]),
                'z': float(self.position[1]),
                'yaw': float(self.theta),
                'timestamp': timestamp
            })

        # 고정 주파수 샘플링
        if self.recording and self.sample_idx < self.clip_length:
            self.sample_accum += dt
            step = 1.0 / float(self.record_hz)
            while self.sample_accum >= step and self.sample_idx < self.clip_length:
                self.traj_buf[self.sample_idx, 0] = self.position[0]
                self.traj_buf[self.sample_idx, 1] = self.position[1]
                self.traj_buf[self.sample_idx, 2] = self.theta
                self.sample_idx += 1
                self.sample_accum -= step

            if self.sample_idx >= self.clip_length:
                self.recording = False
                print(f"Recording completed: {self.clip_length} samples at {self.record_hz} Hz")

    def start_recording(self):
        self.recording = True
        self.trajectory.clear()
        self.sample_idx = 0
        self.sample_accum = 0.0
        # t=0 첫 샘플
        self.traj_buf[self.sample_idx, 0] = self.position[0]
        self.traj_buf[self.sample_idx, 1] = self.position[1]
        self.traj_buf[self.sample_idx, 2] = self.theta
        self.sample_idx += 1
        print(f"🔴 Recording started: target {self.clip_length} samples @ {self.record_hz} Hz")

    def stop_recording(self):
        self.recording = False
        print(f"🟡 Recording stopped ({len(self.trajectory)} trajectory points, {self.sample_idx} buffer samples)")

    def clear_trajectory(self):
        self.recording = False
        self.trajectory.clear()
        self.traj_buf[:] = 0.0
        self.sample_idx = 0
        self.sample_accum = 0.0
        self.position[:] = 0.0
        self.theta = 0.0
        print("🗑️ Cleared. Reset to origin.")

    def save_trajectory(self, filename=None):
        if not self.trajectory:
            print("저장할 궤적이 없습니다.")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.trajectory, f, indent=2)
            
        print(f"📁 궤적이 {filename}에 저장되었습니다.")
        return filename

    # 드로잉
    def draw_grid(self):
        for x in range(0, self.width, self.grid_size):
            color = self.LIGHT_GRAY if x == self.center_x else self.GRAY
            pygame.draw.line(self.screen, color, (x, 50), (x, self.height), 1)
        for y in range(50, self.height, self.grid_size):
            color = self.LIGHT_GRAY if y == self.center_y else self.GRAY
            pygame.draw.line(self.screen, color, (0, y), (self.width, y), 1)
        
        # 축 강조
        pygame.draw.line(self.screen, self.WHITE, (self.center_x, 50), (self.center_x, self.height), 2)
        pygame.draw.line(self.screen, self.WHITE, (0, self.center_y), (self.width, self.center_y), 2)
        
        # OpenGL 좌표계 라벨
        font = pygame.font.Font(None, 16)
        self.screen.blit(font.render("+X", True, self.WHITE), (self.width - 25, self.center_y - 20))
        self.screen.blit(font.render("-X", True, self.WHITE), (5, self.center_y - 20))
        self.screen.blit(font.render("+Z", True, self.WHITE), (self.center_x + 5, 55))  # 화면 위
        self.screen.blit(font.render("-Z", True, self.WHITE), (self.center_x + 5, self.height - 25))  # 화면 아래
        
        pygame.draw.circle(self.screen, self.RED, (self.center_x, self.center_y), 3)

    def draw_button(self, rect, text, color=(200, 200, 200)):
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, self.WHITE, rect, 2)
        text_surf = self.font.render(text, True, self.BLACK)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw_character(self):
        pixel_pos = self.world_to_pixel(self.position)
        
        # 캐릭터 (파란색 원)
        pygame.draw.circle(self.screen, (77, 153, 255), pixel_pos, 12)
        pygame.draw.circle(self.screen, self.WHITE, pixel_pos, 12, 2)
        
        # 방향 화살표 (간단하게)
        arrow_len = 20
        end_x = pixel_pos[0] + arrow_len * math.sin(self.theta)
        end_y = pixel_pos[1] - arrow_len * math.cos(self.theta)  # Z축이 위쪽이므로 -cos
        pygame.draw.line(self.screen, self.RED, pixel_pos, (int(end_x), int(end_y)), 3)
        pygame.draw.circle(self.screen, self.RED, (int(end_x), int(end_y)), 4)

    def draw_trajectory_line(self):
        """JSON 궤적 (노란색)"""
        if len(self.trajectory) < 2:
            return
        
        pts = []
        for point in self.trajectory:
            world_pos = np.array([point['x'], point['z']])
            pts.append(self.world_to_pixel(world_pos))
        
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, self.YELLOW, False, pts, 3)

    def draw_buffer_trajectory(self):
        """고정 주파수 버퍼 궤적 (하얀색)"""
        if self.sample_idx >= 2:
            pts = [self.world_to_pixel(self.traj_buf[i, :2]) for i in range(self.sample_idx)]
            pygame.draw.lines(self.screen, self.WHITE, False, pts, 2)

    def draw(self):
        self.screen.fill(self.BLACK)
        self.draw_grid()

        # 버튼
        start_color = (100, 255, 100) if not self.recording and self.sample_idx < self.clip_length else (200, 200, 200)
        self.draw_button(self.start_button, "Start", start_color)
        self.draw_button(self.clear_button, "Clear")
        done_color = (100, 255, 100) if self.sample_idx == self.clip_length else (200, 200, 200)
        self.draw_button(self.done_button, "Done", done_color)

        # 상태 메시지
        if self.recording:
            msg = f"🔴 Recording... {self.sample_idx}/{self.clip_length} @ {self.record_hz}Hz"
            color = self.RED
        elif self.sample_idx == self.clip_length:
            msg = "✅ Trajectory ready! Click Done or press T to save."
            color = self.GREEN
        else:
            msg = "WASD=relative move, Arrows=absolute, Mouse=look, SPACE/R=record, T=save, C=clear, ESC=exit"
            color = self.WHITE
        self.screen.blit(self.font.render(msg, True, color), (310, 15))

        # 궤적 그리기
        self.draw_trajectory_line()      # 노란색 (JSON)
        self.draw_buffer_trajectory()    # 하얀색 (numpy 버퍼)
        
        # 캐릭터
        self.draw_character()
        
        # 정보 표시
        info_y = 50
        yaw_deg = -math.degrees(self.theta)
        info_texts = [
            f"Pos: ({self.position[0]:.1f}, {self.position[1]:.1f})",
            f"Yaw: {yaw_deg:.1f}°",
            f"JSON pts: {len(self.trajectory)}",
            f"Buffer: {self.sample_idx}/{self.clip_length}"
        ]
        for i, text in enumerate(info_texts):
            surf = pygame.font.Font(None, 20).render(text, True, self.WHITE)
            self.screen.blit(surf, (10, info_y + i * 20))

        pygame.display.flip()

    def get_trajectory(self) -> Optional[np.ndarray]:
        if self.sample_idx == self.clip_length:
            return self.traj_buf.copy()
        return None

    def run(self) -> Optional[np.ndarray]:
        clock = pygame.time.Clock()
        running = True
        
        print("=== 조작법 ===")
        print("WASD: 캐릭터 방향 기준 이동 (W=앞으로, S=뒤로)")
        print("방향키: 절대 방향 이동")
        print("마우스: 바라보는 방향")
        print("Shift: 빠르게, Ctrl: 느리게")
        print("R 또는 SPACE: 녹화 시작/중지")
        print("T: JSON 궤적 저장")
        print("C: 초기화")
        print("ESC: 종료")
        print("===============")
        
        while running:
            dt = clock.tick(60) / 1000.0
            running = self.handle_events()
            self.update(dt)
            self.draw()
            
        pygame.quit()
        return self.get_trajectory()

# 궤적 변환 함수
def to_relative_trajectory(absolute_traj: np.ndarray) -> np.ndarray:
    if absolute_traj is None or absolute_traj.shape[0] == 0:
        return None
    T = absolute_traj.shape[0]
    rel_traj = np.zeros_like(absolute_traj)
    rel_traj[0] = [0.0, 0.0, 0.0]
    for i in range(1, T):
        rel_traj[i, :2] = absolute_traj[i, :2] - absolute_traj[i-1, :2]
        delta_theta = absolute_traj[i, 2] - absolute_traj[i-1, 2]
        # 각도 차이 normalize: -pi ~ pi
        delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
        rel_traj[i, 2] = delta_theta
    return rel_traj

# 사용 예시
if __name__ == "__main__":
    drawer = TrajectoryDrawer(
        clip_length=180,
        record_hz=60,
        world_scale=1.0,
        walk_speed=50.0,   # 3D와 동일
        run_speed=100.0,
        crouch_speed=25.0
    )
    absolute_trajectory = drawer.run()

    if absolute_trajectory is not None:
        print(f"Absolute trajectory shape: {absolute_trajectory.shape}")
        print("First 5 absolute points (OpenGL coords, yaw in rad):")
        for i, p in enumerate(absolute_trajectory[:5]):
            print(f"  {i}: X={p[0]:.6f}, Z={p[1]:.6f}, Yaw={p[2]:.6f}")

        # 상대량으로 변환
        relative_trajectory = to_relative_trajectory(absolute_trajectory)

        # 저장 (3D 호환)
        pos_scale = 1.0  # 3D 코드 단위와 맞춤
        abs_out = to_unit(absolute_trajectory, pos_scale)
        rel_out = to_unit(relative_trajectory, pos_scale)

        torch.save(torch.from_numpy(abs_out).float(), "absolute_trajectory_3d_compatible.pt")
        torch.save(torch.from_numpy(rel_out).float(), "relative_trajectory_3d_compatible.pt")
        print("✅ Saved trajectories compatible with 3D OpenGL code")

    else:
        print("No trajectory created!")