import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from spatialgeometry import Sphere, Cuboid

# -----------------------------------------------------------------------------------#
class Glassbot(DHRobot3D):
    def __init__(self):
        """
        Glassbot Robot using DH model + STL meshes.
        Inherits from DHRobot3D for easy 3D visualization.
        Adds gripper support with green fingers.
        """
        # DH links
        links = self._create_DH()

        # Mesh names
        link3D_names = dict(
            link0="link0",
            link1="link1",
            link2="link2",
            link3="link3",
            link4="link4",
            link5="link5",
            link6="link6"
        )

        # Reference test joint config
        qtest = [0, 0, 0, 0, 0, 0]

        # Alignment transforms for meshes
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.rpy2tr(0, 0, 0, order="xyz"),
            spb.transl(0, 0, 0.291) @ spb.rpy2tr(0, 0, -pi/2, order="xyz"),
            spb.transl(0, 0, 0.564) @ spb.rpy2tr(pi/2, pi, pi/2, order="xyz"),
            spb.transl(0, 0, 0.636) @ spb.rpy2tr(0, pi/2, pi, order="xyz"),
            spb.transl(0.297, 0, 0.636) @ spb.rpy2tr(-pi/2, 0, -pi/2, order="xyz"),
            spb.transl(0.298, 0, 0.638) @ spb.rpy2tr(0, pi/2, 0, order="xyz"),
            spb.transl(0.365, 0, 0.637) @ spb.rpy2tr(0, pi/2, 0, order="xyz"),
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "assets")

        super().__init__(
            links,
            link3D_names,
            name="Glassbot",
            link3d_dir=link3d_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        self.q = qtest
        self._env = None

        # --------------------- Gripper setup --------------------- #
        self._finger_length = 0.08
        self._finger_thickness = 0.01
        self._finger_height = 0.08
        self._finger_gap_open = 0.06
        self._finger_gap_closed = 0.055
        self._finger_gap = self._finger_gap_open
        self._finger_back_offset = 0.02
        self._finger_z_offset = -0.03

        self._left_finger = Cuboid([self._finger_length, self._finger_thickness, self._finger_height],
                                   color=[0.2, 0.8, 0.2, 1])
        self._right_finger = Cuboid([self._finger_length, self._finger_thickness, self._finger_height],
                                    color=[0.2, 0.8, 0.2, 1])

    # --------------------- Add to environment --------------------- #
    def add_to_env(self, env):
        super().add_to_env(env)
        self._env = env
        env.add(self._left_finger)
        env.add(self._right_finger)
        self._update_fingers()

    # --------------------- Finger transforms --------------------- #
    def _finger_transform(self, tcp, side, gap):
        y = gap / 2.0 if side == "left" else -gap / 2.0
        return tcp * SE3(-self._finger_back_offset, y, self._finger_z_offset) * SE3.Ry(pi)

    def _update_fingers(self, q=None):
        if q is None:
            q = self.q
        tcp = self.fkine(q)
        self._left_finger.T = self._finger_transform(tcp, "left", self._finger_gap).A
        self._right_finger.T = self._finger_transform(tcp, "right", self._finger_gap).A

    # --------------------- Gripper control --------------------- #
    def gripper_open(self, steps=20):
        for g in np.linspace(self._finger_gap, self._finger_gap_open, steps):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(0.02)

    def gripper_close(self, steps=20):
        for g in np.linspace(self._finger_gap, self._finger_gap_closed, steps):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(0.02)

    # --------------------- Existing DH --------------------- #
    def _create_DH(self):
        a     = [0.0,   0.270, 0.070, 0.0, 0.0, 0.0]
        d     = [0.290, 0.0,   0.0,   0.302, 0.0, 0.072]
        alpha = [-pi/2, 0.0,  -pi/2,  pi/2, -pi/2, 0.0]
        offset = [0.0, -pi/2, 0.0, 0.0, 0.0, pi]
        qlim = [[-pi, pi], [-pi/2, 110*pi/180], [-110*pi/180, 70*pi/180],
                [-160*pi/180, 160*pi/180], [-120*pi/180, 120*pi/180], [-400*pi/180, 400*pi/180]]
        links = []
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i],
                                        offset=offset[i], qlim=qlim[i]))
        return links

    # --------------------- Test methods --------------------- #
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.add_to_env(env)
        self.gripper_open()
        fig = self.plot(self.q, limits=[-1, 1, -1, 1, 0, 1.2])
        fig._add_teach_panel(self, self.q)
        while plt.fignum_exists(fig.fig.number):
            self.add_to_env(env)
            env.step(0.02)
            fig.step(0.02)
            plt.pause(0.02)
            self.gripper_close()
        env.hold()
        fig.hold()

    def test_with_fk(self, q_goal, tol=0.005):
        env = swift.Swift()
        env.launch(realtime=False)
        self.q = self._qtest
        self.add_to_env(env)
        T_desired = self.fkine(q_goal)
        qtraj = rtb.jtraj(self.q, q_goal, 30).q
        for q in qtraj:
            self.q = q
            env.step(0.0001)
        T_final = self.fkine(self.q)
        pos_err = np.linalg.norm(T_final.t - T_desired.t)
        if pos_err <= tol:
            print(f"Reached goal within {pos_err:.4f} m")
        else:
            print(f"❌ Goal off by {pos_err:.4f} m")
        env.hold()

# -----------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = Glassbot()
    r.test()
