from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np
from pyglm import glm
import random

colors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
]
vertices = [
    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]
]
normals = [
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0]
]

def random_color(min_val=0.3, max_val=1.0):
    """
    검은 계열을 피해서 랜덤한 RGB 컬러를 생성합니다.
    :param min_val: 최소 채널값 (0.3 이상 추천)
    :param max_val: 최대 채널값 (1.0 이하)
    :return: (R, G, B) float 튜플
    """
    r = random.uniform(min_val, max_val)
    g = random.uniform(min_val, max_val)
    b = random.uniform(min_val, max_val)
    return (r, g, b)

def blend_color(colora, colorb):
    return tuple((a + b) / 2.0 for a, b in zip(colora, colorb))



def draw_colored_cube(size_x, size_y=-1, size_z=-1, color=None):
    """
    면을 구분하거나 지정된 색상으로 칠한 cube를 생성하는 함수입니다.
    :param size_x: cube의 x길이
    :param size_y: cube의 y길이
    :param size_z: cube의 z길이
    :param color: (R, G, B) float 튜플, 지정된 색상으로 전체 면을 칠함
    """
    if (size_y < 0): size_y = size_x
    if (size_z < 0): size_z = size_x
    glPushMatrix()
    glScaled(size_x, size_y, size_z)
    glBegin(GL_QUADS)
    for i in range(6):
        if color:
            glColor3fv(color)
        else:
            glColor3fv(colors[i])
        glNormal3fv(normals[i])
        for j in range(4):
            glVertex3fv(vertices[i * 4 + j])
    glEnd()
    glPopMatrix()


def draw_colored_sphere(radius):
    """
    Quadric object인 sphere을 생성하기 위한 함수입니다.
    :param radius: sphere의 크기
    """
    quadric = gluNewQuadric()  # Quadric object 생성 (http://www.gisdeveloper.co.kr/?p=35)
    glPushMatrix()
    glColor3fv((1.0, 0.0, 0.0))
    gluSphere(quadric, radius, 30, 30)
    glPopMatrix()
    gluDeleteQuadric(quadric)  # 다 쓰고 이렇게 삭제해줘야 되더라


def draw_axes(grid_size = 500, step = 10):
    """
    축과 격자를 그리기 위한 함수입니다.
    grid_size를 미리 정의해서 그 길이만큼 격자랑 XYZ axis 그리게 해두었습니다.
    축의 길이가 너무 짧아 길게 설정하였는데 그만큼 Clipping distance도 늘렸습니다
    TODO: CLippind distance랑 연계하기.
    """

    glLineWidth(2.0)

    # Draw XYZ axes
    glBegin(GL_LINES)

    # X-axis (Red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(-grid_size, 0, 0)
    glVertex3f(grid_size, 0, 0)

    # Y-axis (Green)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, -grid_size, 0)
    glVertex3f(0, grid_size, 0)

    # Z-axis (Blue)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, 0, -grid_size)
    glVertex3f(0, 0, grid_size)

    glEnd()

    # 격자 그리기 위해 줄 크기 줄이기
    glLineWidth(1.0)

    # 격자 색
    glColor3f(0.5, 0.5, 0.5)
    glBegin(GL_LINES)

    for i in range(-grid_size, grid_size + 1, step):
        # X방향
        glVertex3f(i, 0, -grid_size)
        glVertex3f(i, 0, grid_size)

        # Z방향
        glVertex3f(-grid_size, 0, i)
        glVertex3f(grid_size, 0, i)

    glEnd()


def set_lights():
    """
    기본 enviroment의 빛을 조절하는 함수입니다.
    """
    # 조명 설정
    glEnable(GL_LIGHTING)  # 조명 활성화
    glEnable(GL_LIGHT0)  # 기본 조명 활성화
    glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화

    # 조명 파라미터 설정
    ambient_light = [0.5, 0.5, 0.5, 1.0]  # 주변광
    diffuse_light = [0.2, 0.2, 0.2, 1.0]  # 난반사광
    r = 0.5
    specular_light = [r, r, r, 1.0]  # 정반사광
    position = [5.0, 5.0, 5.0, 0.0]  # 조명 위치

    # 조명 속성 적용
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)
    glLightfv(GL_LIGHT0, GL_POSITION, position)

    # 재질(Material) 설정
    glEnable(GL_COLOR_MATERIAL)  # 재질 색상 활성화
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)  # 재질 색상 설정
    specular_reflection = [r, r, r, 1.0]
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_reflection)  # 정반사 재질
    glMateriali(GL_FRONT, GL_SHININESS, 1)  # 광택 설정


def rotation_between_vectors(v1, v2):

    v1 = glm.normalize(v1)
    v2 = glm.normalize(v2)
    cos_theta = glm.dot(v1, v2)

    # ★★★★★ 180도 회전 예외 처리 추가 ★★★★★
    if cos_theta < -0.99999:
        rotation_axis = glm.cross(glm.vec3(1.0, 0.0, 0.0), v1)
        if glm.length2(rotation_axis) < 1e-6:
            rotation_axis = glm.cross(glm.vec3(0.0, 1.0, 0.0), v1)
        
        rotation_axis = glm.normalize(rotation_axis)

        return glm.angleAxis(glm.pi(), rotation_axis)

    rotation_axis = glm.cross(v1, v2)
    s = glm.sqrt((1 + cos_theta) * 2)
    invs = 1 / (s + 1e-9)
    return glm.quat(s * 0.5,
                    rotation_axis.x * invs,
                    rotation_axis.y * invs,
                    rotation_axis.z * invs)


def bone_rotation(forward):
    """
    Skeleton 구조에서 뼈와 뼈사이의 회전을 구하는 함수입니다.
    :param forward: 전방벡터값
    :return: 회전행렬
    """
    originalDir = glm.vec3(0, 1, 0)
    if glm.length2(glm.cross(originalDir, forward)) < 1e-6:
        if glm.dot(originalDir, forward) < 0:
            rot = glm.angleAxis(glm.pi(), glm.vec3(0, 1, 0))
            rot = glm.angleAxis(glm.pi(), glm.vec3(1, 0, 0))
        else:
            rot = glm.quat(1, 0, 0, 0)
    else:
        rot = rotation_between_vectors(forward, originalDir)
    return rot

def draw_undercircle(radius=1.0):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)  # 채워진 스타일로 설정
    gluDisk(quadric, 0.0, radius, 32, 1)
    gluDeleteQuadric(quadric)

def draw_arrow(circle_radius, arrow_length, color):

    R = np.eye(3)
    forward = R @ np.array([0,0,-1])
    forward[1] = 0

    dir = forward / (np.linalg.norm(forward) + 1e-20)
    tail = -dir * circle_radius
    head = tail - dir * arrow_length
    
    glColor3fv(color)
    
    glBegin(GL_LINES)
    glVertex3f(tail[0], 0, tail[2])
    glVertex3f(head[0], 0, head[2])
    glEnd()

    perp = np.array([-dir[2], 0, dir[0]])
    arrow_head_width = arrow_length * 0.3  # 화살촉 너비 (조정 가능)
    # 화살촉의 base 위치 (라인의 head에서 약간 뒤쪽)
    arrow_head_base = head + dir * (arrow_length * 0.5)
    left = arrow_head_base + perp * arrow_head_width
    right = arrow_head_base - perp * arrow_head_width

    glBegin(GL_TRIANGLES)
    glVertex3f(head[0], 0, head[2])
    glVertex3f(left[0], 0, left[2])
    glVertex3f(right[0], 0, right[2])
    glEnd()