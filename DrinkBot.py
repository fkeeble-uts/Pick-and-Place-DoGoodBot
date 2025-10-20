
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D 
import numpy as np
import os
from math import pi

# Robot class
class DrinkBot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()
        link3D_names = dict(
            link0='CRB15000_Joint0', color0=(0.4, 0.45, 0.5, 1),
            link1='CRB15000_Joint1', color1=(0.4, 0.45, 0.5, 1),
            link2='CRB15000_Joint2', color2=(0.4, 0.45, 0.5, 1),
            link3='CRB15000_Joint3', color3=(0.4, 0.45, 0.5, 1),
            link4='CRB15000_Joint4', color4=(0.8, 0.8, 0.8, 1),
            link5='CRB15000_Joint5', color5=(0.8, 0.8, 0.8, 1),
            link6='CRB15000_Joint6', color6=(0.8, 0.8, 0.8, 1),
            link7='CRB15000_Joint6', color7=(0.8, 0.8, 0.8, 1)
        )
        qtest = [0, pi/2, 0, 0, pi, 0]
        qtest_transforms = [spb.transl(0, 0, 0),
                            spb.transl(0.0625, -0.1375, 0.2141),
                            spb.transl(0.211875, -0.1512, 0.317),
                            spb.transl(0.2263, -0.0115, 1.0385),
                            spb.transl(0.3895, 0.048, 1.161),
                            spb.transl(0.8, 0.0143, 1.161),
                            spb.transl(0.996, 0.0565, 1.2425),
                            spb.transl(0.996, 0.0565, 1.2425)] @ spb.transl(-0.1433, -0.101, 0)
        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "3D Robot Assets")
        super().__init__(links, link3D_names, name='CRB15000', link3d_dir=link3d_path, qtest=qtest, qtest_transforms=qtest_transforms)
        
        qlim_deg = np.array([
            [-360, -27, -85, -144, -180, -360],
            [360,  208,  240,  127,  270,  360]
        ])
        self.qlim = np.deg2rad(qlim_deg)

        self.home_q = [-pi/2, 0, 0, 0, 0, 0]
        self.q = self.home_q

    def _create_DH(self):
        links = []
        d = [0.399, -0.0863, -0.0863, 0.636, 0.0085, 0.101, 0]
        a = [0.15, -0.706, -0.110, 0, 0.0805, 0, 0]
        alpha = [-pi/2, pi, -pi/2, -pi/2, -pi/2, 0, 0]
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i]))
        return links