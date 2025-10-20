import time
import numpy as np
import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from math import pi
from spatialgeometry import Cuboid
from roboticstoolbox import models


class Serverbot:
    def __init__(self):
        # Load standard UR3 from robotics toolbox
        self.robot = rtb.models.UR3()
        
        # Gripper configuration
        self._finger_gap = 0.035
        self._finger_open = 0.035
        self._finger_closed = 0.014
        self._left_finger = Cuboid(scale=[0.1, 0.01, 0.05], color="green")
        self._right_finger = Cuboid(scale=[0.1, 0.01, 0.05], color="green")
        self._env = None

    # ----------------------------- GRIPPER CONTROL -----------------------------
    @property
    def q(self):
        return self.robot.q
    
    @q.setter
    def q(self, value):
        self.robot.q = value
    
    @property
    def base(self):
        return self.robot.base
    
    @base.setter
    def base(self, value):
        self.robot.base = value

    def add_to_env(self, env):
        env.add(self.robot, readonly=False)
        self._env = env
        env.add(self._left_finger)
        env.add(self._right_finger)
        self._update_fingers()

    def _update_fingers(self):
        tcp = self.robot.fkine(self.robot.q)
        gap = self._finger_gap
        self._left_finger.T = (tcp * SE3(0, gap + 0.005, 0) * SE3.Ry(pi)).A
        self._right_finger.T = (tcp * SE3(0, -gap - 0.005, 0) * SE3.Ry(pi)).A

    def open_gripper(self, speed=0.02):
        for g in np.linspace(self._finger_gap, self._finger_open, 20):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(speed)

    def close_gripper(self, speed=0.02):
        for g in np.linspace(self._finger_gap, self._finger_closed, 20):
            self._finger_gap = g
            self._update_fingers()
            if self._env:
                self._env.step(speed)

    # ----------------------------- TESTING -----------------------------
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)
        self.add_to_env(env)

        q_start = self.robot.q
        q_goal = [q_start[i] + pi / 6 for i in range(len(q_start))]
        traj = rtb.jtraj(q_start, q_goal, 40).q

        for q in traj:
            self.robot.q = q
            self._update_fingers()
            env.step(0.02)

        self.close_gripper()
        time.sleep(1)
        self.open_gripper()
        print("Sim Successful")
        env.hold()


if __name__ == "__main__":
    robot = Serverbot()
    robot.test()