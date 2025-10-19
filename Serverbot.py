import os
import time
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from math import pi
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialgeometry import Cuboid


class Serverbot(DHRobot3D):
    def __init__(self):
        # ----------------------------- DH MODEL -----------------------------
        links = self._create_DH()

        # Mesh mapping for UR3 only (6 links + 1 tool frame)
        # Removed 'base rail ur3' entry completely
        link3D_names = dict(
            link0="slider_rail_ur3",  # static base mount
            link1="shoulder_ur3",
            link2="upperarm_ur3",
            link3="forearm_ur3",
            link4="wrist1_ur3",
            link5="wrist2_ur3",
            link6="wrist3_ur3",  # wrist3 mesh also used for TCP
        )

        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "assets")

        # Reference configuration (6 DOF)
        qtest = [0, -pi / 2, 0, 0, 0, 0]

        # 7 transforms (rail mount + 6 robot segments)
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.rpy2tr(0, 0, -pi/2, order="xyz"),  # linear_rail_mount (static)
            spb.transl(0, 0.146, 0) @ spb.rpy2tr(0, pi, pi, order="xyz"),  # shoulder
            spb.transl(0, 0.146, -0.13) @ spb.rpy2tr(0, 0, -pi / 2, order="xyz"),  # upperarm
            spb.transl(0, 0.39, -0.0378) @ spb.rpy2tr(0, 0, -pi / 2, order="xyz"),  # forearm
            spb.transl(0, 0.603, -0.0378) @ spb.rpy2tr(0, pi / 2, -pi / 2, order="xyz"),  # wrist1
            spb.transl(0, 0.603, -0.1225) @ spb.rpy2tr(0, pi / 2, -pi / 2, order="xyz"),  # wrist2
            spb.transl(0.08535, 0.603, -0.1225) @ spb.rpy2tr(0, pi / 2, -pi / 2, order="xyz"),  # wrist3/TCP
        ]

        super().__init__(
            links,
            link3D_names,
            name="Serverbot",
            link3d_dir=link3d_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms,
        )

        # Rotate base so robot stands upright on the table
        self.base = self.base * SE3.Rx(pi / 2) * SE3.Ry(pi / 2)
        self.q = qtest

        # ----------------------------- GRIPPER -----------------------------
        self._finger_gap = 0.035
        self._finger_open = 0.035
        self._finger_closed = 0.014
        self._left_finger = Cuboid(scale=[0.1, 0.01, 0.05], color="green")
        self._right_finger = Cuboid(scale=[0.1, 0.01, 0.05], color="green")
        self._env = None

    # ----------------------------------------------------------------------
    def _create_DH(self):
        """Standard 6DOF UR3 DH model."""
        a = [0, -0.24365, -0.21325, 0, 0, 0]
        d = [0.146, 0, 0, 0.121, 0.083, 0.0819]
        alpha = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
        offset = [0, 0, 0, 0, 0, 0]
        qlim = [[-2 * pi, 2 * pi] for _ in range(6)]

        links = [
            rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i])
            for i in range(6)
        ]
        return links

    # ----------------------------------------------------------------------
    def add_to_env(self, env):
        super().add_to_env(env)
        self._env = env
        env.add(self._left_finger)
        env.add(self._right_finger)
        self._update_fingers()

    # ----------------------------------------------------------------------
    def _update_fingers(self):
        tcp = self.fkine(self.q)
        gap = self._finger_gap
        self._left_finger.T = (tcp * SE3(0, gap + 0.005, 0) * SE3.Ry(pi)).A
        self._right_finger.T = (tcp * SE3(0, -gap - 0.005, 0) * SE3.Ry(pi)).A

    # ----------------------------------------------------------------------
    def open_gripper(self, speed=0.02):
        for g in np.linspace(self._finger_gap, self._finger_open, 20):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(speed)

    # ----------------------------------------------------------------------
    def close_gripper(self, speed=0.02):
        for g in np.linspace(self._finger_gap, self._finger_closed, 20):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(speed)

    # ----------------------------------------------------------------------
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)

        q_start = self.q
        q_goal = [q_start[i] + pi / 6 for i in range(len(q_start))]
        traj = rtb.jtraj(q_start, q_goal, 40).q

        for q in traj:
            self.q = q
            self._update_fingers()
            env.step(0.02)

        self.close_gripper()
        time.sleep(1)
        self.open_gripper()
        print("Sim Successful")
        env.hold()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    robot = Serverbot()
    robot.test()