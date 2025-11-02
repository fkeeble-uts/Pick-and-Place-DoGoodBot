import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D 
import numpy as np
import os
from math import pi

class IngredientBot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()
        link3D_names = dict(
            link0='TM5_Basem', color0=(0.1, 0.1, 0.15, 1),
            link1='TM5_Link1m', color1=(0.8, 0.8, 0.8, 1),
            link2='TM5_Link2new', color2=(0.8, 0.8, 0.8, 1),
            link3='TM5_Link3new', color3=(0.8, 0.8, 0.8, 1),
            link4='TM5_Link4new', color4=(0.8, 0.8, 0.8, 1),
            link5='TM5_Link5new', color5=(0.0, 0.6, 0.8, 1),
            link6='TM5_EndEff', color6=(0.0, 0.6, 0.8, 1),
            link7='TM5_EndEff', color7=(0.0, 0.6, 0.8, 1)
        )
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [
                            SE3.Rz(pi/2).A @ spb.transl(0, 0, 0),
                            SE3.Rz(pi/2).A @ spb.transl(-0.038, 0, 0.066),
                            SE3.Rz(pi/2).A @ spb.transl(-0.207, -0.06, 0.1465),
                            SE3.Rz(pi/2).A @ spb.transl(-0.0666, -0.0445, 0.4759),
                            SE3.Rz(pi/2).A @ spb.transl(-0.17, -0.0452, 0.727),
                            SE3.Rz(pi/2).A @ spb.transl(-0.1831, 0.0005, 0.872),
                            SE3.Rz(pi/2).A @ spb.transl(-0.24, -0.0225, 0.893),
                            spb.transl(5, 0, 0)] 
        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "3D Robot Assets")
        super().__init__(links, link3D_names, name='TM5', link3d_dir=link3d_path, qtest=qtest, qtest_transforms=qtest_transforms)
        
        qlim_deg = np.array([
            [-270, -107, -155, -180, -180, -270],
            [270, 107, 155, 180, 180, 270]
        ])
        self.qlim = np.deg2rad(qlim_deg)

        self.home_q = [pi, 0, 0, 0, 0, 0]
        self.q = self.home_q

    def _create_DH(self):
        links = []
        d = [0.1452, 0, 0, 0.124, -0.1066, 0.125, 0]
        a = [0, -0.329, -0.3115, 0, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0, 0]
        offset = [0, -pi/2, 0, pi/2, 0, 0, 0]
        for i in range(6): 
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i]))
        return links                