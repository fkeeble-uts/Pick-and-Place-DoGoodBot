import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import numpy as np
import matplotlib.pyplot as plt
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
            [-360, -25, -360, -360, -360, -360],  # Min limits in degrees
            [360,  205,  360,  360,  360,  360]   # Max limits in degrees
        ])
        self.qlim = np.deg2rad(qlim_deg) # Convert to radians and assign

        self.q = qtest

    def _create_DH(self):
        links = []
        d = [0.399, -0.0863, -0.0863, 0.636, 0.0085, 0.101, 0]
        a = [0.15, -0.706, -0.110, 0, 0.0805, 0, 0]
        alpha = [-pi/2, pi, -pi/2, -pi/2, -pi/2, 0, 0]
        qlim = [[-2*pi, 2*pi] for _ in range(6)]
        for i in range(6):
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i]))
        return links

    # --- NEW AND IMPROVED find_ikine METHOD ---
    def find_ikine(self, target_tr, initial_q_guess=None):
        num_attempts = 15
        min_limits = self.qlim[0, :]
        max_limits = self.qlim[1, :]

        for i in range(num_attempts):
            # Use the provided guess on the first attempt, then switch to random guesses
            if i == 0 and initial_q_guess is not None:
                q_guess = initial_q_guess
            else:
                # Generate a new random guess that is guaranteed to be within joint limits
                q_guess = np.random.uniform(low=min_limits, high=max_limits)

            # Solve for the inverse kinematics
            ik_result = self.ikine_LM(target_tr, q0=q_guess)

            # Check if the solver was successful AND if the solution is within our limits
            if ik_result.success:
                solution = ik_result.q
                
                # Check if all joint angles in the solution are within the defined qlim
                if np.all((solution >= min_limits) & (solution <= max_limits)):
                    # Valid solution found, return it
                    return solution
        
        # If the loop completes without finding a valid solution
        logging.warning(
            f"Failed to find an IK solution within joint limits for target pose after {num_attempts} attempts."
        )
        return self.q # Return the current joint state to avoid errors
        
    def animate(self, sim_env, start_q, end_q, steps, held_ingredient=None, relative_tf=None):
        q_path = rtb.jtraj(start_q, end_q, steps).q
        for q_config in q_path:
            self.q = np.clip(q_config, self.qlim[0, :], self.qlim[1, :])
            sim_env.step(0.02)

            if held_ingredient and relative_tf:
                ee_transform = self.fkine(self.q)
                held_ingredient.base = ee_transform * relative_tf
                held_ingredient.q = np.zeros(1)

    def execute_full_sequence(self):
        sim_env = swift.Swift()
        sim_env.launch(realtime=True)
        self.add_to_env(sim_env)

        # q Guesses
        q_guesses = {
            "initial": np.deg2rad([90, 200, -120, -180, -50, -180]),
            "pickup": np.deg2rad([90, 200, -130, -100, -40, -180]),
            "towall": np.deg2rad([0, 200, -100, -40, -180, -180]),
            "from_dropoff": np.deg2rad([0, 200, -90, -50, -180, -180]),
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

        # Set up ingredients
        ingredient_pickup_poses = [SE3(-0.1, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.15, 0.47, 0) * SE3.Rz(pi/2),
                              SE3(-0.25, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.3, 0.47, 0) * SE3.Rz(pi/2),
                              SE3(-0.4, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.45, 0.47, 0) * SE3.Rz(pi/2),
                              SE3(-0.55, 0.4, 0) * SE3.Rz(pi/2), SE3(-0.6, 0.47, 0) * SE3.Rz(pi/2),
                              SE3(-0.7, 0.4, 0) * SE3.Rz(pi/2)] @ spb.transl(0.5, 0, 0)
        ingredient_placement_poses = [SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                 SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2)] @ spb.transl(-0.5, 0, -ee_offset)

        sim_ingredients = []
        for i, tr in enumerate(ingredient_pickup_poses):
            ingredient_obj = SceneObject(f'ingredient_{i+1}')
            ingredient_obj.q = np.array([0])
            ingredient_obj.base = tr
            ingredient_obj.add_to_env(sim_env)
            sim_ingredients.append(ingredient_obj)

        logging.info("Moving to starting pose...")
        start_q = self.find_ikine(WAYPOINTS["pickup"], initial_q_guess=q_guesses["initial"])
        self.animate(sim_env, self.q, start_q, 50)
        
        ee_transform = self.fkine(self.q)
        q_degrees = np.round(np.rad2deg(self.q), 2)
        logging.info(f"End effector transform at step: {ee_transform}")
        logging.info(f"Joint state at step (deg): {q_degrees}")

        for i in range(len(ingredient_pickup_poses)):
            logging.info(f"Now moving ingredient {i+1}")

            base_pickup_tr = SE3(SE3(ingredient_pickup_poses[i]).t[0], SE3(ingredient_pickup_poses[i]).t[1], ingredient_height - ee_offset)
            hover_pickup_tr = base_pickup_tr * SE3(0, 0, hover_height) * SE3.Rx(pi)
            
            logging.info("Moving to hover position")
            pickup_hover_q = self.find_ikine(hover_pickup_tr, q_guesses["pickup"])
            self.animate(sim_env, self.q, pickup_hover_q, 50)
            
            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")
            
            logging.info("Moving over ingredient")
            grasp_tr = base_pickup_tr * SE3.Rx(pi)
            grasp_path = rtb.ctraj(hover_pickup_tr, grasp_tr, 50)
            for pose in grasp_path:
                self.q = self.find_ikine(pose)
                sim_env.step(0.02)
            
            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")
            
            gripped_ingredient = sim_ingredients[i]
            ee_tr = self.fkine(self.q)
            relative_tr = SE3(np.linalg.inv(ee_tr.A)) * gripped_ingredient.base
            
            logging.info(f"Now gripping ingredient {i+1}...")
            self.animate(sim_env, self.q, pickup_hover_q, 50, gripped_ingredient, relative_tr)
            
            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")
            
            logging.info("Moving to wall")
            via_q_1 = self.find_ikine(WAYPOINTS["pickup"])
            self.animate(sim_env, self.q, via_q_1, 50, gripped_ingredient, relative_tr)
            via_q_2 = self.find_ikine(WAYPOINTS["middle"], q_guesses["from_initial"])
            self.animate(sim_env, self.q, via_q_2, 50, gripped_ingredient, relative_tr)
            via_q_3 = self.find_ikine(WAYPOINTS["towall"], q_guesses["from_dropoff"])
            self.animate(sim_env, self.q, via_q_3, 50, gripped_ingredient, relative_tr)

            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")

            stack_level = 0
            if i >= 6: stack_level = 4.4 * stack_height
            elif i >= 3: stack_level = 2.15 * stack_height

            base_placement_tr = SE3(SE3(ingredient_placement_poses[i]).t[0], SE3(ingredient_placement_poses[i]).t[1],
                                    SE3(ingredient_placement_poses[i]).t[2] + stack_level + ingredient_height)
            hover_placement_tr = base_placement_tr * SE3(0, 0, hover_height) * SE3.Rx(pi)
            
            logging.info("Moving to wall hover position")
            placement_hover_q = self.find_ikine(hover_placement_tr, q_guesses["towall"])
            self.animate(sim_env, self.q, placement_hover_q, 50, gripped_ingredient, relative_tr)
            
            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")

            logging.info("Placing the ingredient down")
            final_placement_tr = base_placement_tr * SE3.Rx(pi)
            placement_path = rtb.ctraj(hover_placement_tr, final_placement_tr, 50)
            for pose in placement_path:
                self.q = self.find_ikine(pose)
                sim_env.step(0.02)
                ee_tr = self.fkine(self.q)
                gripped_ingredient.base = ee_tr * relative_tr
                gripped_ingredient.q = np.zeros(1)

            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")
            
            gripped_ingredient.base = base_placement_tr
            logging.info(f"Ingredient {i+1} has placed")
            
            logging.info("Moving back")
            self.animate(sim_env, self.q, via_q_3, 50)
            self.animate(sim_env, self.q, via_q_2, 50)
            self.animate(sim_env, self.q, via_q_1, 50)
            
            ee_transform = self.fkine(self.q)
            q_degrees = np.round(np.rad2deg(self.q), 2)
            logging.info(f"End effector transform at step: {ee_transform}")
            logging.info(f"Joint state at step (deg): {q_degrees}")

        logging.info("Completed wall construction")
        sim_env.close()

class SceneObject(DHRobot3D):
    def __init__(self, name):
        links = [rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[0, 0])]
        model_name = 'HalfSizedRedGreenBrick'
        link3D_names = dict(link0=model_name, link1=model_name)
        qtest = [0]
        qtest_transforms = [SE3(0, 0, 0).A, SE3(0, 0, 0).A]
        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name=name, link3d_dir=current_path, qtest=qtest, qtest_transforms=qtest_transforms)

if __name__ == "__main__":
    robot_arm = IngredientBot()
    robot_arm.execute_full_sequence()