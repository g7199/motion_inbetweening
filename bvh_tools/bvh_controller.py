from pyglm import glm
from bvh_tools.virtual_transforms import get_pelvis_virtual_safe
from bvh_tools.joint import Joint, VirtualRootJoint
import math
import numpy as np

def mat4_close(a, b, eps=1e-4):
    for i in range(3):
        for j in range(3):
            if abs(a[i][j] - b[i][j]) > eps:
                print(f"Mismatch at ({i},{j}): {a[i][j]} vs {b[i][j]}")
                return False
    return True

class MotionFrame:
    def __init__(self):
        self.joint_rotations = {}
        self.joint_positions = {}


class Motion:
    def __init__(self, frames, frame_time):
        self.frames = frames
        self.frame_time = frame_time
        self.motion_data = []
        self.quaternion_frames = []

    def get_frames(self):
        return self.frames
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            new_motion = Motion(0, self.frame_time)
            new_motion.quaternion_frames = self.quaternion_frames[key]
            new_motion.frames = len(new_motion.quaternion_frames)
            return new_motion
        elif isinstance(key, int):
            return self.quaternion_frames[key]
        else:
            raise TypeError("Motion indices must be integers or slices.")

    def add_frame_data(self, frame_data):
        self.motion_data.append(frame_data)

    def build_quaternion_frames(self, joint_order):
        for frame in self.motion_data:
            motion_frame = MotionFrame()
            channel_index = 0
            for joint in joint_order:
                num_channels = len(joint.channels)
                ch_values = frame[channel_index:channel_index + num_channels]

                rotation = glm.vec3(0.0)
                position = glm.vec3(0.0)

                for i, ch in enumerate(joint.channels):
                    val = ch_values[i]
                    if ch.endswith("position"):
                        if ch == "Xposition": position.x = val
                        elif ch == "Yposition": position.y = val
                        elif ch == "Zposition": position.z = val
                    elif ch.endswith("rotation"):
                        if ch == "Xrotation": rotation.x = glm.radians(val)
                        elif ch == "Yrotation": rotation.y = glm.radians(val)
                        elif ch == "Zrotation": rotation.z = glm.radians(val)

                quat = glm.quat(1, 0, 0, 0)
                for ch in joint.channels:
                    if ch.endswith('rotation'):
                        angle = glm.radians(ch_values[joint.channels.index(ch)])
                        axis = None
                        if ch == 'Xrotation':
                            axis = glm.vec3(1, 0, 0)
                        elif ch == 'Yrotation':
                            axis = glm.vec3(0, 1, 0)
                        elif ch == 'Zrotation':
                            axis = glm.vec3(0, 0, 1)
                        quat = quat * glm.angleAxis(angle, axis)


                motion_frame.joint_rotations[joint.name] = quat
                if "position" in ''.join(joint.channels):
                    motion_frame.joint_positions[joint.name] = position

                channel_index += num_channels
            self.quaternion_frames.append(motion_frame)

    def apply_virtual(self, root):
        vr = VirtualRootJoint(root)

        for frame in self.quaternion_frames:
            hip_name = next(iter(frame.joint_rotations))
            ap = frame.joint_positions.get(hip_name, glm.vec3(0))
            ar = frame.joint_rotations[hip_name]

            ap_local, ar_local = get_pelvis_virtual_safe(ap, ar)
            ap_global = ap - ap_local
            ar_global = ar * glm.conjugate(ar_local)

            frame.joint_positions[hip_name] = ap_local
            frame.joint_rotations[hip_name] = ar_local

            frame.joint_positions["VirtualRoot"] = ap_global
            frame.joint_rotations["VirtualRoot"] = ar_global

        return vr

    def apply_to_skeleton(self, frame_index: int, joint_root: Joint):

        frame = self.quaternion_frames[frame_index]

        def apply(joint: Joint):
            name = joint.name
            rot = frame.joint_rotations.get(name, glm.quat(1, 0, 0, 0))
            R = glm.mat4_cast(rot)

            offset = glm.vec3(joint.offset)
            T_offset = glm.translate(glm.mat4(1.0), offset)

            if name in frame.joint_positions:
                T_root = glm.translate(glm.mat4(1.0), frame.joint_positions[name])
                local_transform = T_root * T_offset * R
            else:
                local_transform = T_offset * R

            joint.kinematics = local_transform

            for child in joint.children:
                apply(child)

        apply(joint_root)


def parse_bvh(filename):
    joints_stack = []
    root_joint = None
    motion = None

    with open(filename, 'r') as file:
        lines = file.readlines()

    line_iter = iter(lines)
    for line in line_iter:
        if 'ROOT' in line or 'JOINT' in line:
            joint_name = line.strip().split()[1]
            next(line_iter)
            offset = [float(x) for x in next(line_iter).strip().split()[1:]]
            channels = next(line_iter).strip().split()[2:]
            joint = Joint(joint_name, offset, channels)

            if joints_stack:
                joints_stack[-1].add_child(joint)
            else:
                root_joint = joint

            joints_stack.append(joint)

        elif 'End Site' in line:
            next(line_iter)
            offset = [float(x) for x in next(line_iter).strip().split()[1:]]
            end_joint = Joint('End Site', offset, [])
            joints_stack[-1].add_child(end_joint)
            next(line_iter)

        elif '}' in line:
            joints_stack.pop()

        elif 'MOTION' in line:
            break

    frames = int(next(line_iter).strip().split()[1])
    frame_time = float(next(line_iter).strip().split()[2])
    motion = Motion(frames, frame_time)

    for _ in range(frames):
        frame_data = [float(val) for val in next(line_iter).strip().split()]
        motion.add_frame_data(frame_data)

    return root_joint, motion


def get_preorder_joint_list(root_joint):
    joint_list = []

    def traverse(joint):
        joint_list.append(joint)
        for child in joint.children:
            traverse(child)

    traverse(root_joint)
    return joint_list

def connect(motion1, motion2, transition_frames=100, start_index_m2=3):
    if abs(motion1.frame_time - motion2.frame_time) > 1e-6:
        raise ValueError("Frame times of the two motions do not match.")
    if transition_frames > motion1.get_frames() or transition_frames > (motion2.get_frames() - start_index_m2):
        raise ValueError("Not enough frames to perform blending with the requested transition_frames.")

    new_motion = Motion(0, motion1.frame_time)

    # 1. offset 계산 (VirtualRoot 기준)
    last_frame_m1 = motion1.quaternion_frames[-1]
    first_frame_m2 = motion2.quaternion_frames[start_index_m2]

    p1 = last_frame_m1.joint_positions["VirtualRoot"]
    p2 = first_frame_m2.joint_positions["VirtualRoot"]

    r1 = last_frame_m1.joint_rotations["VirtualRoot"]
    r2 = first_frame_m2.joint_rotations["VirtualRoot"]
    rotation_offset = r1 * glm.conjugate(r2)
    position_offset = p1 - rotation_offset * p2

    # 2. motion2 복사본 생성 + VirtualRoot offset 적용
    adjusted_motion2 = []

    for i, frame in enumerate(motion2.quaternion_frames):
        new_frame = MotionFrame()
        apply_offset = (i >= start_index_m2)  # start_index_m2부터만 offset 적용

        for joint_name, quat in frame.joint_rotations.items():
            if joint_name == "VirtualRoot" and apply_offset:
                new_frame.joint_rotations[joint_name] = rotation_offset * quat
            else:
                new_frame.joint_rotations[joint_name] = quat

        for joint_name, pos in frame.joint_positions.items():
            if joint_name == "VirtualRoot" and apply_offset:
                new_frame.joint_positions[joint_name] = rotation_offset * pos + position_offset
            else:
                new_frame.joint_positions[joint_name] = pos

        adjusted_motion2.append(new_frame)

    # 3. motion1의 blending 전까지 복사
    for frame in motion1.quaternion_frames[:-transition_frames]:
        new_motion.quaternion_frames.append(frame)

    # 4. blending 구간 생성 (offset 이미 적용된 adjusted_motion2 사용)
    for i in range(transition_frames):
        t = (i + 1) / (transition_frames + 1)
        frame1 = motion1.quaternion_frames[-transition_frames + i]
        frame2 = adjusted_motion2[start_index_m2 + i]

        blended_frame = MotionFrame()
        joint_names = set(frame1.joint_rotations.keys()) | set(frame2.joint_rotations.keys())

        for joint_name in joint_names:
            # rotation slerp
            if joint_name in frame1.joint_rotations and joint_name in frame2.joint_rotations:
                r1 = frame1.joint_rotations[joint_name]
                r2 = frame2.joint_rotations[joint_name]
                blended_frame.joint_rotations[joint_name] = glm.slerp(r1, r2, t)

            # position lerp
            if joint_name in frame1.joint_positions and joint_name in frame2.joint_positions:
                p1 = frame1.joint_positions[joint_name]
                p2 = frame2.joint_positions[joint_name]
                blended_frame.joint_positions[joint_name] = glm.mix(p1, p2, t)

        new_motion.quaternion_frames.append(blended_frame)

    # 5. motion2 transition 이후 프레임 복사 (offset 적용된 버전에서)
    for frame in adjusted_motion2[start_index_m2 + transition_frames:]:
        new_motion.quaternion_frames.append(frame)

    new_motion.frames = len(new_motion.quaternion_frames)
    return new_motion
