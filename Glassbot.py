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
from spatialgeometry import Sphere

# -----------------------------------------------------------------------------------#
class Glassbot(DHRobot3D):
    def __init__(self):
        """
        ABB IRB120 Robot using DH model + STL meshes.
        Inherits from DHRobot3D for easy 3D visualization.
        """
        # DH links
        links = self._create_DH()

        # Mesh names (must exist in the same folder as this file)
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
        qtest = [pi/2, 0, 0, 0, 0, 0]

        # Alignment transforms for meshes (adjust manually until they sit correctly!)
        qtest_transforms = [
            spb.transl(0, 0, 0) @ spb.rpy2tr(0, 0, 0, order="xyz"),        # base
            spb.transl(0, 0, 0.291) @ spb.rpy2tr(0, 0, -pi/2, order="xyz"),     # link1
            spb.transl(0, 0, 0.564) @ spb.rpy2tr(pi/2, pi, pi/2, order="xyz"),  # link2
            spb.transl(0, 0, 0.636) @ spb.rpy2tr(0, pi/2, pi, order="xyz"),  # link3
            spb.transl(0.297, 0, 0.636) @ spb.rpy2tr(-pi/2, 0, -pi/2, order="xyz"),   # link4
            spb.transl(0.298, 0, 0.638) @ spb.rpy2tr(0, pi/2, 0, order="xyz"),  # link5
            spb.transl(0.365, 0, 0.637) @ spb.rpy2tr(0, pi/2, 0, order="xyz"),  # link6 (tool flange)
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "assets")
        # Init parent
        super().__init__(
            links,
            link3D_names,
            name="ABB_IRB120",
            link3d_dir=link3d_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        ABB IRB120 Standard DH parameters
        a [m], d [m], alpha [rad]
        """
        a     = [0.0,   0.270, 0.070, 0.0, 0.0, 0.0]
        d     = [0.290, 0.0,   0.0,   0.302, 0.0, 0.072]
        alpha = [-pi/2, 0.0,  -pi/2,  pi/2, -pi/2, 0.0]
        offset = [0.0, -pi/2, 0.0, 0.0, 0.0, pi]

        qlim = [
            [-180*pi/180,  180*pi/180],   # J1
            [-90*pi/180,   110*pi/180],   # J2
            [-110*pi/180,  70*pi/180],    # J3
            [-160*pi/180,  160*pi/180],   # J4
            [-120*pi/180,  120*pi/180],   # J5
            [-400*pi/180,  400*pi/180]    # J6
        ]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(
                d=d[i], a=a[i], alpha=alpha[i],
                offset=offset[i], qlim=qlim[i]
            )
            links.append(link)
        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3D objects into Swift and plotting with teach panel
        """
        env = swift.Swift()
        env.launch(realtime=True)

        # Map qtest into robot.q and add meshes
        self.q = self._qtest
        self.add_to_env(env)

        # Matplotlib teach panel
        fig = self.plot(self.q, limits=[-1, 1, -1, 1, 0, 1.2])
        fig._add_teach_panel(self, self.q)

        # Keep both running
        while plt.fignum_exists(fig.fig.number):
            self.add_to_env(env)
            env.step(0.02)
            fig.step(0.02)
            plt.pause(0.02)

        env.hold()
        fig.hold()

    def test_with_fk(self, q_goal, tol=0.005):
        """
        Move robot to q_goal and check with fkine if end-effector
        is within tol meters of expected.
        """
        env = swift.Swift()
        env.launch(realtime=False)

        # Start at qtest
        self.q = self._qtest
        self.add_to_env(env)

        # Desired pose from FK
        T_desired = self.fkine(q_goal)

        # Smooth path
        qtraj = rtb.jtraj(self.q, q_goal, 30).q
        for q in qtraj:
            self.q = q
            env.step(0.0001)

        # Final pose
        T_final = self.fkine(self.q)

        # Check position error
        pos_err = np.linalg.norm(T_final.t - T_desired.t)
        if pos_err <= tol:
            print(f"Reached goal within {pos_err:.4f} m")
        else:
            print(f"❌ Goal off by {pos_err:.4f} m")

        env.hold()


# -----------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = Glassbot()
    q_goal = [0.5, -0.3, 0.2, 0.5, 0.5, 0.5]  # user-defined
    r.test_with_fk(q_goal)
