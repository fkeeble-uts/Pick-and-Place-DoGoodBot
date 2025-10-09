import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os
import logging
from math import pi

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Robot class
class IngredientBot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()
        link3D_names = dict(
            link0='CRB15000_Joint1', color0=(0.2, 0.2, 0.2, 1),
            link1='CRB15000_Joint2', color1=(0.1, 0.1, 0.1, 1),
            link2='CRB15000_Joint3', color2=(0.9, 0.9, 0.9, 1),
            link3='CRB15000_Joint4', color3=(0.2, 0.2, 0.2, 1),
            link4='CRB15000_Joint5', color4=(0.2, 0.2, 0.2, 1),
            link5='CRB15000_Joint6', color5=(0.2, 0.2, 0.2, 1),
            link6='CRB15000_Joint7', color6=(0.2, 0.2, 0.2, 1),
            link7='CRB15000_Joint7', color7=(0.2, 0.2, 0.2, 1)
        )
        qtest = [0, 0, -pi/2, 0, 0, pi, 0]
        qtest_transforms = [spb.transl(0, 0, 0) @ spb.rpy2tr(0, 0, -pi/2, order='xyz'),
                            spb.transl(0.0061, -0.00008, 0.0137) @ spb.trotx(-pi/2),
                            spb.transl(0.01405, -0.0001, 0.0151) @ spb.rpy2tr(0, 0, -pi/2, order='xyz'),
                            spb.transl(0.014, -0.0001, 0.0012) @ spb.rpy2tr(0, 0, -pi/2, order='xyz'),
                            spb.transl(0.0139, -0.0001, -0.00463) @ spb.rpy2tr(0, 0, -pi/2, order='xyz'),
                            spb.transl(0, 0, 0) @ spb.rpy2tr(0, pi/2, -pi/2, order='xyz'),
                            spb.transl(0, 0, 0) @ spb.rpy2tr(0, pi/2, -pi/2, order='xyz'),
                            spb.transl(0, 0, 0) @ spb.rpy2tr(0, pi/2, -pi/2, order='xyz')]
        # rotation: robot y = global z, robot z = global y, robot x = global x
        # translation: robot x = global y, robot y = global z, robot z = global x
        current_path = os.path.abspath(os.path.dirname(__file__))

        super().__init__(
            links,
            link3D_names,
            name='CRB15000',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        # Scale all visuals
        link_scale = 0.0001  # adjust as needed
        try:
            visuals = []
            for attr_name in ["_link3d", "links_3d", "link_3d", "_links_3d"]:
                if hasattr(self, attr_name):
                    attr_value = getattr(self, attr_name)
                    if isinstance(attr_value, dict):
                        visuals = list(attr_value.values())
                    elif isinstance(attr_value, list):
                        visuals = attr_value

            for visual in visuals:
                if hasattr(visual, "scale"):
                    visual.scale = [link_scale] * 3
        except Exception as e:
            logging.warning(f"Could not scale visuals: {e}")

        # Base orientation
        self.base = self.base * SE3.Rx(pi/2) * SE3.Ry(pi/2)
        self.q = qtest

    def _create_DH(self):
        links = [rtb.PrismaticDH(theta=pi, a=0, alpha=pi/2, qlim=[-0.8, 0])]
        a = [-0.15, -0.24365, -0.21325, 0, 0, 0]
        d = [0.4, 0, 0, 0.121, 0.083, 0.0819]
        alpha = [-pi/2, 0, 0, pi/2, -pi/2, 0]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
        return links

if __name__ == "__main__":
    # Launch Swift and add robot
    env = swift.Swift()
    env.launch(realtime=True)

    robot_arm = IngredientBot()
    robot_arm.add_to_env(env)

    # Set a reasonable camera view
    env.set_camera_pose([1.5, 1.5, 1.0], [0, 0, 0])

    # Keep the simulation running
    while True:
        env.step(0.02)
