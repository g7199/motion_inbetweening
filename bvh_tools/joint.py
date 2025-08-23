from pyglm import glm

class Joint:
    def __init__(self, name, offset, channels):
        self.name = name
        self.offset = offset
        self.channels = channels
        self.children = []
        self.parent = None
        self.kinematics = glm.mat4(1.0)

    def add_child(self, child_joint):
        child_joint.parent = self
        self.children.append(child_joint)


class VirtualRootJoint(Joint):
    def __init__(self, root):
        super().__init__("VirtualRoot", [0, 0, 0], ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Yrotation', 'Xrotation'])
        self.add_child(root)