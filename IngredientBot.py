import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os
from math import pi
import numpy as np

# Robot class
class IngredientBot(DHRobot3D):
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
        qtest = [0, 0, pi/2, 0, 0, pi]
        qtest_transforms = [spb.transl(0, 0, 0),
                            spb.transl(0.0625, -0.1375, 0.2141),
                            spb.transl(0.211875, -0.1512, 0.317),
                            spb.transl(0.2263, -0.0115, 1.0385),
                            spb.transl(0.3895, 0.048, 1.161),
                            spb.transl(0.8, 0.0143, 1.161),
                            spb.transl(0.996, 0.056, 1.242),
                            spb.transl(0.996, 0.056, 1.242)] @ spb.transl(-0.14385, -0.102, 0)

        current_path = os.path.abspath(os.path.dirname(__file__))

        super().__init__(
            links,
            link3D_names,
            name='CRB15000',
            link3d_dir=current_path,
            qtest=qtest,
            qtest_transforms=qtest_transforms
        )

        self.q = qtest

    def _create_DH(self):
        links = []
        a = [0.15, -0.707, -0.110, 0, 0.08, 0, 0]
        d = [0.4, -0.0863, -0.0863, 0.637, 0, 0.101, 0]
        alpha = [-pi/2, pi, -pi/2, -pi/2, -pi/2, 0, 0]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
        return links
    
    # Slider function
    def slider_callback(self, value, joint_index):
        """
        Callback function for the Swift sliders.
        """
        # Create a copy of the current joint configuration
        new_q = self.q.copy()
        
        # Modify the specific joint angle in the copy
        new_q[joint_index] = np.deg2rad(value)
        
        # Reassign the joint angle
        self.q = new_q


if __name__ == "__main__":
    import time

    # Launch Swift environment
    env = swift.Swift()
    env.launch(realtime=True)

    # Initialize robot and add to environment
    robot_arm = IngredientBot()
    robot_arm.add_to_env(env)

    # Create sliders
    sliders = []
    for i in range(6):
        initial_q_deg = np.rad2deg(robot_arm.q[i])
        slider = swift.Slider(
            cb=lambda value, j=i: robot_arm.slider_callback(value, j),
            min=-180,
            max=180,
            step=1,
            value=initial_q_deg,
            desc=f'Joint {i} Angle',
            unit='°'
        )
        sliders.append(slider)

    for s in sliders:
        env.add(s)

    print("Loaded robot and sliders")
    
    # Main loop to keep the simulation running
    while True:
        env.step(0.02)
        time.sleep(0.02)