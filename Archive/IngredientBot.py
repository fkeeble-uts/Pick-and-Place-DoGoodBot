import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import numpy as np
import os
import logging
from math import pi
import time

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
        super().__init__(links, link3D_names, name='CRB15000', link3d_dir=current_path, qtest=qtest, qtest_transforms=qtest_transforms)
        
        qlim_deg = np.array([
            [-360, -180, -85, -180, -180, -360],
            [360,  180,  225,  180,  180,  360]
        ])
        self.qlim = np.deg2rad(qlim_deg)

        self.q = qtest

    def _create_DH(self):
        links = []
        d = [0.399, -0.0863, -0.0863, 0.636, 0.0085, 0.101, 0]
        a = [0.15, -0.706, -0.110, 0, 0.0805, 0, 0]
        alpha = [-pi/2, pi, -pi/2, -pi/2, -pi/2, 0, 0]
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i]))
        return links

    def find_ikine(self, target_tr, initial_q_guess=None, ignore_rotation=False):
        num_attempts = 15
        min_limits = self.qlim[0, :]
        max_limits = self.qlim[1, :]
        mask = [1, 1, 1, 0, 0, 0] if ignore_rotation else None

        for i in range(num_attempts):
            if i == 0 and initial_q_guess is not None:
                q_guess = initial_q_guess
            else:
                q_guess = np.random.uniform(low=min_limits, high=max_limits)

            ik_result = self.ikine_LM(target_tr, q0=q_guess, mask=mask)

            if ik_result.success:
                solution = ik_result.q
                if np.all((solution >= min_limits) & (solution <= max_limits)):
                    return solution
        
        logging.warning(
            f"Failed to find an IK solution within joint limits for target pose after {num_attempts} attempts."
        )
        return self.q
        
    def animate(self, sim_env, start_q, end_q, steps, held_ingredient=None, relative_tf=None):
        q_path = rtb.jtraj(start_q, end_q, steps).q
        for q_config in q_path:
            self.q = np.clip(q_config, self.qlim[0, :], self.qlim[1, :])
            sim_env.step(0.02)

            if held_ingredient and relative_tf:
                ee_transform = self.fkine(self.q)
                held_ingredient.base = ee_transform * relative_tf

    def slider_callback(self, value, joint_index):
        new_q = self.q.copy()
        new_q[joint_index] = np.deg2rad(value)
        self.q = new_q
        
        q_degrees = np.round(np.rad2deg(self.q), 2)
        print(f"Joint state updated (deg): {q_degrees}")
        
        tr_matrix = self.fkine(self.q) 
        print(f"Resulting robot pose:\n{np.round(tr_matrix.A, 4)}")

    def execute_full_sequence(self, sim_env, sim_ingredients, ingredient_pickup_poses):
        q_guesses = {
            "initial": np.deg2rad([-50, 54, 184, 21, 113, -2]),
            "pickup": np.deg2rad([-82, 31, 165, 180, -134, 95]),
            "towall": np.deg2rad([69, 29, 155, 180, -126, 95]),
            "from_dropoff": np.deg2rad([44, 75, 203, 180, -132, 95]),
            "from_initial": np.deg2rad([-90, 200, -90, -50, -180, -180])
        }
        WAYPOINTS = {
            "pickup": SE3(-0.4, 0.4, 0.4) * SE3.Rx(pi),
            "towall": SE3(-0.4, -0.4, 0.5) * SE3.Rx(pi),
            "middle": SE3(-0.8, 0, 0.4) * SE3.Rx(pi)
        }
        ingredient_height = 0.30
        ee_offset = 0.265
        hover_height = 0.1
        stack_height = 0.015
        
        ingredient_placement_poses = [SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2)] @ spb.transl(-0.5, 0, -ee_offset)

        logging.info("Moving to starting pose...")
        start_q = self.find_ikine(WAYPOINTS["pickup"], initial_q_guess=q_guesses["initial"], ignore_rotation=True)
        self.animate(sim_env, self.q, start_q, 50)
        
        for i in range(len(ingredient_pickup_poses)):
            logging.info(f"--- Now moving ingredient {i+1} ---")

            base_pickup_tr = SE3(SE3(ingredient_pickup_poses[i]).t[0], SE3(ingredient_pickup_poses[i]).t[1], ingredient_height - ee_offset)
            hover_pickup_tr = base_pickup_tr * SE3(0, 0, hover_height) * SE3.Rx(pi)
            
            logging.info("Moving to hover position")
            pickup_hover_q = self.find_ikine(hover_pickup_tr, q_guesses["pickup"], ignore_rotation=True)
            self.animate(sim_env, self.q, pickup_hover_q, 50)
            
            logging.info("Moving over ingredient")
            grasp_tr = base_pickup_tr * SE3.Rx(pi)
            grasp_path = rtb.ctraj(hover_pickup_tr, grasp_tr, 50)
            for pose in grasp_path:
                self.q = self.find_ikine(pose, initial_q_guess=self.q)
                sim_env.step(0.02)
            
            gripped_ingredient = sim_ingredients[i]
            ee_tr = self.fkine(self.q)
            relative_tr = SE3(np.linalg.inv(ee_tr.A)) * gripped_ingredient.base
            
            logging.info("Gripping ingredient...")
            self.animate(sim_env, self.q, pickup_hover_q, 50, gripped_ingredient, relative_tr)
            
            logging.info("Moving to wall")
            via_q_1 = self.find_ikine(WAYPOINTS["pickup"], ignore_rotation=True)
            via_q_2 = self.find_ikine(WAYPOINTS["middle"], q_guesses["from_initial"], ignore_rotation=True)
            via_q_3 = self.find_ikine(WAYPOINTS["towall"], q_guesses["from_dropoff"], ignore_rotation=True)
            
            self.animate(sim_env, self.q, via_q_1, 50, gripped_ingredient, relative_tr)
            self.animate(sim_env, self.q, via_q_1, 50, gripped_ingredient, relative_tr)
            self.animate(sim_env, self.q, via_q_2, 50, gripped_ingredient, relative_tr)
            self.animate(sim_env, self.q, via_q_3, 50, gripped_ingredient, relative_tr)

            stack_level = 0
            if i >= 6: stack_level = 2 * (2.15 * stack_height)
            elif i >= 3: stack_level = 2.15 * stack_height

            base_placement_tr = SE3(SE3(ingredient_placement_poses[i]).t[0], SE3(ingredient_placement_poses[i]).t[1],
                                    SE3(ingredient_placement_poses[i]).t[2] + stack_level + ingredient_height)
            hover_placement_tr = base_placement_tr * SE3(0, 0, hover_height) * SE3.Rx(pi)
            
            logging.info("Moving to wall hover position")
            placement_hover_q = self.find_ikine(hover_placement_tr, q_guesses["towall"])
            self.animate(sim_env, self.q, placement_hover_q, 50, gripped_ingredient, relative_tr)
            
            logging.info("Placing the ingredient down")
            final_placement_tr = base_placement_tr * SE3.Rx(pi)
            placement_path = rtb.ctraj(hover_placement_tr, final_placement_tr, 50)
            for pose in placement_path:
                self.q = self.find_ikine(pose, initial_q_guess=self.q)
                sim_env.step(0.02)
                ee_tr = self.fkine(self.q)
                gripped_ingredient.base = ee_tr * relative_tr
            
            gripped_ingredient.base = final_placement_tr
            logging.info(f"Ingredient {i+1} has been placed")
            
            logging.info("Moving back")
            self.animate(sim_env, self.q, placement_hover_q, 50)
            self.animate(sim_env, self.q, via_q_3, 50)
            self.animate(sim_env, self.q, via_q_2, 50)
            self.animate(sim_env, self.q, via_q_1, 50)

        logging.info("Completed wall construction")


class SceneObject(DHRobot3D):
    def __init__(self, name):
        links = [rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[0,0])]
        model_name = 'HalfSizedRedGreenBrick'
        link3D_names = dict(link0=model_name, link1=model_name)
        qtest = [0]
        qtest_transforms = [SE3(0, 0, 0).A, SE3(0, 0, 0).A]
        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name=name, link3d_dir=current_path, qtest=qtest, qtest_transforms=qtest_transforms)


def setup_scene(env, poses):
    ingredients = []
    for i, tr in enumerate(poses):
        ingredient_obj = SceneObject(f'ingredient_{i+1}')
        ingredient_obj.base = tr
        ingredient_obj.add_to_env(env)
        ingredients.append(ingredient_obj)
    return ingredients


if __name__ == "__main__":
    # Decides if automatic movement or teach mode will be used
    RUN_AUTOMATED_SEQUENCE = False

    env = swift.Swift()
    env.launch(realtime=True)
    robot_arm = IngredientBot()
    robot_arm.add_to_env(env)
    
    ingredient_pickup_poses = [
        SE3(-0.1, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.15, 0.47, 0) * SE3.Rz(pi/2),
        SE3(-0.25, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.3, 0.47, 0) * SE3.Rz(pi/2),
        SE3(-0.4, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.45, 0.47, 0) * SE3.Rz(pi/2),
        SE3(-0.55, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.6, 0.47, 0) * SE3.Rz(pi/2),
        SE3(-0.7, 0.4, 0) * SE3.Rz(pi/2)
    ] @ spb.transl(0.5, 0, 0)
    
    sim_ingredients = setup_scene(env, ingredient_pickup_poses)

    if RUN_AUTOMATED_SEQUENCE:
        print("Running automated pick-and-place sequence...")
        robot_arm.execute_full_sequence(env, sim_ingredients, ingredient_pickup_poses)
        print("Sequence complete.")
        print("Press Ctrl+C in terminal to exit.")
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
    else:
        print("Running in interactive mode with sliders...")
        sliders = []
        for i in range(6):
            initial_q_deg = np.rad2deg(robot_arm.q[i])
            slider = swift.Slider(
                cb=lambda value, j=i: robot_arm.slider_callback(value, j),
                min=-270,
                max=270,
                step=1,
                value=initial_q_deg,
                desc=f'Joint {i} Angle',
                unit='°'
            )
            sliders.append(slider)

        for s in sliders:
            env.add(s)

        print("Loaded robot, bricks, and sliders. Press Ctrl+C in terminal to exit.")
        while True:
            try:
                env.step(0.02)
                time.sleep(0.02)
            except KeyboardInterrupt:
                break
    
    env.close()
    print("Program finished.")