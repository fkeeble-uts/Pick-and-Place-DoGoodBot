
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D 
import numpy as np
import os
import logging
from math import pi

# ----------------------------------------------------
# I. INITIAL SETUP AND CONFIGURATION
# ----------------------------------------------------

# Log config (can be moved to main_sim.py, but keeping here for class logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# ----------------------------------------------------
# II. ROBOT CLASS DEFINITION
# ----------------------------------------------------

class Drinkbot(DHRobot3D):
    def __init__(self):
        links = self._create_DH()
        link3D_names = dict(
            link0='TM5_Basem', color0=(0.4, 0.45, 0.5, 1),
            link1='TM5_Link1m', color1=(0.4, 0.45, 0.5, 1),
            link2='TM5_Link2new', color2=(0.4, 0.45, 0.5, 1),
            link3='TM5_Link3new', color3=(0.4, 0.45, 0.5, 1),
            link4='TM5_Link4new', color4=(0.8, 0.8, 0.8, 1),
            link5='TM5_Link5new', color5=(0.8, 0.8, 0.8, 1),
            link6='TM5_EndEffector', color6=(0.8, 0.8, 0.8, 1),
            link7='TM5_EndEffector', color7=(0.8, 0.8, 0.8, 1)
        )
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [
                            SE3.Rz(pi/2).A @ spb.transl(0, 0, 0),
                            SE3.Rz(pi/2).A @ spb.transl(-0.038, 0, 0.066),
                            SE3.Rz(pi/2).A @ spb.transl(-0.207, -0.06, 0.1465),
                            SE3.Rz(pi/2).A @ spb.transl(-0.0666, -0.0445, 0.4759),
                            SE3.Rz(pi/2).A @ spb.transl(-0.17, -0.0452, 0.727),
                            SE3.Rz(pi/2).A @ spb.transl(-0.1831, 0.0005, 0.872),
                            SE3.Rz(pi/2).A @ spb.transl(-0.2915, 0.0005, 0.9058),
                            spb.transl(5, 0, 0)] 
        current_path = os.path.abspath(os.path.dirname(__file__))
        link3d_path = os.path.join(current_path, "assets")
        super().__init__(links, link3D_names, name='TM5', link3d_dir=link3d_path, qtest=qtest, qtest_transforms=qtest_transforms)
        
        qlim_deg = np.array([
            [-270, -180, -155, -180, -180, -270],
            [270, 180, 155, 180, 180, 270]
        ])
        self.qlim = np.deg2rad(qlim_deg)

        self.q = qtest

    # ----------------------------------------------------
    # III. KINEMATICS IMPLEMENTATION (DH Parameters)
    # ----------------------------------------------------

    def _create_DH(self):
        links = []
        d = [0.1452, 0, 0, 0.124, -0.1066, 0.125, 0]
        a = [0, -0.329, -0.3115, 0, 0, 0, 0]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0, 0]
        offset = [0, -pi/2, 0, pi/2, 0, 0, 0]
        for i in range(6): 
            links.append(rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i]))
        return links

    # ----------------------------------------------------
    # IK. HELPER FUNCTIONS
    # ----------------------------------------------------

    def find_ikine(self, target_tr, initial_q_guess=None, ignore_rotation=False):
        num_attempts = 15
        min_limits = self.qlim[0, :]
        max_limits = self.qlim[1, :]
        # Note: If mask is None, full pose (position + orientation) is used
        mask = [1, 1, 1, 0, 0, 0] if ignore_rotation else None 

        for i in range(num_attempts):
            if i == 0 and initial_q_guess is not None:
                q_guess = initial_q_guess
            else:
                q_guess = np.random.uniform(low=min_limits, high=max_limits)

            ik_result = self.ikine_LM(target_tr, q0=q_guess, mask=mask, joint_limits=True)

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
        
        # This part of the calculation seems to aim for 9 poses in a grid
        ingredient_placement_poses = ([SE3(-0.3, -0.4, 0) * SE3.Rz(pi/2), SE3(-0.434, -0.4, 0) * SE3.Rz(pi/2),
                                     SE3(-0.568, -0.4, 0) * SE3.Rz(pi/2)] * 3) @ spb.transl(-0.5, 0, -ee_offset)

        logging.info("Moving to starting pose...")
        start_q = self.find_ikine(WAYPOINTS["pickup"], initial_q_guess=q_guesses["initial"], ignore_rotation=True)
        self.animate(sim_env, self.q, start_q, 50)
        
        for i in range(len(ingredient_pickup_poses)):
            logging.info(f"--- Now moving ingredient {i+1} ---")

            # Calculations for pickup pose
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
            self.animate(sim_env, self.q, via_q_2, 50, gripped_ingredient, relative_tr)
            self.animate(sim_env, self.q, via_q_3, 50, gripped_ingredient, relative_tr)

            # Calculations for placement pose
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
            
            # Release ingredient
            gripped_ingredient.base = final_placement_tr
            logging.info(f"Ingredient {i+1} has been placed")
            
            logging.info("Moving back")
            # The held_ingredient/relative_tf arguments are omitted for movement without the brick
            self.animate(sim_env, self.q, placement_hover_q, 50)
            self.animate(sim_env, self.q, via_q_3, 50)
            self.animate(sim_env, self.q, via_q_2, 50)
            self.animate(sim_env, self.q, via_q_1, 50)

        logging.info("Completed wall construction")