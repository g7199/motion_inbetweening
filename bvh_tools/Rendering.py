from OpenGL.GL import *
from pyglm import glm
import numpy as np
from bvh_tools.utils import draw_colored_cube, draw_colored_sphere, bone_rotation, draw_arrow, draw_undercircle

joint_size = 3

#OpenGL_accelerate 사용하면 numpy로 변환해줘야함.
def glm_mat4_to_glf(m: glm.mat4) -> np.ndarray:
    return np.array(m.to_list(), dtype=np.float32).flatten()

def draw_humanoid(root_joint, color):
    """
    Skeleton을 그리기 위한 함수입니다.
    :param root_joint: 그릴 joint (전역 kinematics가 이미 계산되어 있음)
    :param color: RGB 컬러 (tuple 또는 list of 3 floats)
    """
    glPushMatrix()
    glMultMatrixf(glm_mat4_to_glf(root_joint.kinematics))
    draw_joint(root_joint.children[0], color)
    glPopMatrix()

def draw_joint(joint, color):
    """
    Joint를 그리기 위한 함수입니다.
    전역 좌표계의 kinematics를 그대로 적용하고, 관절이면 sphere, 아니라면 뼈대를 그립니다.
    :param color: RGB 컬러
    """
    glPushMatrix()
    glMultMatrixf(glm_mat4_to_glf(joint.kinematics))
    
    if joint.name != "joint_Root":
        draw_colored_sphere(joint_size)
    
    for child in joint.children:
        if joint.name != "joint_Root":
            draw_bone(child.offset, color)
        draw_joint(child, color)
    
    glPopMatrix()

def draw_bone(offset, color):
    """
    Skeleton에서 뼈를 그리기 위한 함수입니다.
    :param offset: 뼈 길이를 구하기 위한 값
    :param color: RGB 컬러
    """
    mid = [offset[0] / 2.0, offset[1] / 2.0, offset[2] / 2.0]
    rot_quat = bone_rotation(glm.vec3(*offset))
    rot_mat = glm.mat4_cast(rot_quat)
    glPushMatrix()
    glTranslatef(*mid)
    glMultMatrixf(np.array(rot_mat, dtype=np.float32).flatten())
    glScalef(joint_size, abs(glm.length(glm.vec3(*offset)) - 2 * joint_size) / 2, joint_size / 3)
    draw_colored_cube(1, color=color)
    glPopMatrix()

def draw_virtual_root_axis(kinematics, color, circle_radius=10, arrow_length=20):
    """
    root Transform에서 조그만한 3차원 축을 그리기 위함입니다.
    virtual root의 위치를 받아 회전만큼 회전하여 pelvis의 회전을 시각적으로 확인합니다.
    """
    glPushMatrix()
    glMultMatrixf(glm_mat4_to_glf(kinematics))
    draw_arrow(circle_radius, arrow_length, color)
    glRotatef(90, 1.0, 0.0, 0.0)
    glColor3f(1.0, 1.0, 1.0)
    draw_undercircle(10)
    glPopMatrix()

def draw_trajectory(trajectory_vel, traj_mean, traj_std, dt=1/60.0, color=(0.0, 1.0, 0.0), line_width=2.0):
    """
    [T, 2] 형태의 trajectory_vel (각 행: [vel_x, vel_z])를 누적 position으로 변환해 땅바닥(y=0)에 초록색 선으로 그립니다.
    :param trajectory_vel: [T, 2] 배열 (velocity)
    :param dt: timestep (기본: 1/60)
    :param color: 선 색상 (기본: 초록색)
    :param line_width: 선 두께 (기본: 2.0)
    """
    if len(trajectory_vel) == 0:
        return  # 빈 trajectory 무시

    # trajectory_vel를 numpy 배열로 변환 (리스트일 경우 대비)
    trajectory_vel = np.array(trajectory_vel)
    trajectory_vel = trajectory_vel * traj_std + traj_mean

    # Velocity 누적해서 position 계산 (첫 pos=[0,0])
    trajectory_pos = np.cumsum(trajectory_vel, axis=0)  # [T, 2]
    trajectory_pos = np.insert(trajectory_pos, 0, [0.0, 0.0], axis=0)  # 첫 프레임 [0,0] 추가

    glPushMatrix()
    glLineWidth(line_width)  # 선 두께 설정
    glColor3f(*color)        # 색상 설정 (기본: 초록색)
    
    glBegin(GL_LINE_STRIP)   # 연속된 선 그리기 시작
    for point in trajectory_pos:
        x, z = point
        glVertex3f(x, 0.0, z)  # y=0으로 땅바닥에 고정
    glEnd()                  # 그리기 종료
    
    glPopMatrix()

def draw_positions_points_frame(positions_frame, pos_mean=None, pos_std=None,
                                point_size=4.0, color=(0.0, 1.0, 0.0),
                                use_sphere=False, sphere_radius=2.5):
    """
    positions_frame: [J, 3] 현재 프레임의 3D 좌표들
    pos_mean/pos_std가 주어지면 복원, 없으면 그대로 사용
    """
    if positions_frame is None or len(positions_frame) == 0:
        return

    pos = np.asarray(positions_frame, dtype=np.float32)
    if pos_mean is not None and pos_std is not None:
        pos = pos * np.asarray(pos_std, dtype=np.float32) + np.asarray(pos_mean, dtype=np.float32)

    if not use_sphere:
        glPointSize(point_size)
        glColor3f(*color)
        glBegin(GL_POINTS)
        for j in range(pos.shape[0]):
            x, y, z = map(float, pos[j])
            glVertex3f(x, y, z)
        glEnd()
    else:
        glColor3f(*color)
        for j in range(pos.shape[0]):
            x, y, z = map(float, pos[j])
            glPushMatrix()
            glTranslatef(x, y, z)
            draw_colored_sphere(sphere_radius)
            glPopMatrix()

