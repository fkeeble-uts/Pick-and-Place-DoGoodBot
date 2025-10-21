import numpy as np
import roboticstoolbox as rtb
import logging
from math import pi
from spatialmath import SE3

class RobotController:
    def __init__(self, env, scene):
        self.env = env
        self.scene = scene
        self.carried_object = None
        self.grasp_transform = None
        self.carried_object_name = None
        self.attachments = {}   

    def attach_objects(self, parent_obj, child_obj):
        if parent_obj is child_obj:
            print("Cannot attach an object to itself")
            return 
        # Find relative transform for object being attached
        T_world_parent = SE3(parent_obj.T)
        T_world_child = SE3(child_obj.T)
        T_rel = T_world_parent.inv() @ T_world_child
        # store the attachement 
        self.attachments.setdefault(parent_obj, []).append((child_obj, T_rel))
        print("Attatched {child_obj.name} to {parent_obj.name}")

    def pickup_object(self, robot, obj_to_pickup):
        if self.carried_object is None:
            T_world_tcp = SE3(robot.fkine(robot.q).A)
            T_world_object = SE3(obj_to_pickup.T)
            self.grasp_transform = T_world_tcp.inv() * T_world_object
            self.carried_object = obj_to_pickup
            object_type = type(obj_to_pickup).__name__
            self.carried_object_name = f"{object_type}"
            print(f"[{robot.name}] Picked up object ({self.carried_object_name}).")
        else:
            print("Already carrying an object.")

    def drop_object(self, robot):
        """'Detaches' the object from the robot and prints its final location."""
        if self.carried_object is not None:
            final_pose = SE3(self.carried_object.T)
            final_position = np.round(final_pose.t, 3)
            parent_name = self.carried_object_name     
            self.carried_object = None
            self.grasp_transform = None
            self.carried_object_name = None

            has_attachments = self.carried_object in self.attachments
            children_list = " (and attached objects)" if has_attachments else ""
            print(f"[{robot.name}] Dropped '{parent_name}'{children_list} at position: {final_position}")
        else:
             print(f"[{robot.name}] Not carrying any object to drop.")

    def _update_carried_object_pose(self, robot):
        if self.carried_object is not None:
            T_world_tcp = robot.fkine(robot.q)
            T_world_new_parent = T_world_tcp @ self.grasp_transform
            self.carried_object.T = T_world_new_parent.A
            
            if self.carried_object in self.attachments:
                for child_obj, T_rel in self.attachments[self.carried_object]:
                    T_world_new_child = T_world_new_parent @ T_rel
                    child_obj.T = T_world_new_child.A


    def find_ikine(self, robot, target_tr, initial_q_guess=None, ignore_var="", ignore_rotation=False, hover_max=None):
        num_attempts = 300
        min_limits = robot.qlim[0, :]
        max_limits = robot.qlim[1, :]
        mask = [1, 1, 1, 1, 1, 1]
        coord_map = {"x": 0, "y": 1, "z": 2}
        if ignore_var in coord_map: mask[coord_map[ignore_var]] = 0
        if ignore_rotation: mask[3:6] = [0, 0, 0]
        for i in range(num_attempts):
            if i == 0 and initial_q_guess is not None: q_guess = initial_q_guess
            else: q_guess = np.random.uniform(low=min_limits, high=max_limits)
            ik_result = robot.ikine_LM(target_tr, q0=q_guess, mask=mask)
            if ik_result.success:
                solution = ik_result.q
                if np.all((solution >= min_limits) & (solution <= max_limits)):
                    if hover_max is not None and ignore_var in coord_map:
                        axis_index = coord_map[ignore_var]
                        solution_pose = robot.fkine(solution)
                        solution_coord = solution_pose.t[axis_index]
                        target_coord = target_tr.t[axis_index]
                        lower_bound = min(target_coord, target_coord + hover_max)
                        upper_bound = max(target_coord, target_coord + hover_max)
                        if lower_bound <= solution_coord <= upper_bound:
                            print(f"IK solution found on attempt {i+1} with valid hover on axis '{ignore_var}'.")
                            return solution, True
                    else:
                        return solution, True
        logging.warning(f"Failed to find a valid IK solution after {num_attempts} attempts.")
        return robot.q, False

    def move_cartesian(self, robot, start_q, target_pose, num_steps, ignore_rotation=False):
        print(f"[{robot.name}] Following Cartesian path...")
        start_pose = robot.fkine(start_q)
        cartesian_path = rtb.ctraj(start_pose, target_pose, num_steps)
        current_q = start_q.copy()
        for i, next_pose in enumerate(cartesian_path):
            q_step, solved = self.find_ikine(robot, next_pose, initial_q_guess=current_q, ignore_rotation=ignore_rotation)
            if solved:
                robot.q = q_step
                current_q = q_step
                self._update_carried_object_pose(robot)
                self.env.step(self.scene.SIM_STEP_TIME)
            else:
                logging.warning(f"IK failed at step {i+1}/{num_steps} during Cartesian move. Halting motion.")
                return robot.q
        print(f"✓ Cartesian path complete.")
        return robot.q

    def wrap_to_near(self, q_goal, q_ref):
        return q_ref + (q_goal - q_ref + pi) % (2 * pi) - pi

    def move_to_q(self, robot, q_target, steps=None, name=""):
        if steps is None: steps = self.scene.TRAJ_STEPS
        q_start = robot.q.copy()
        q_target = self.wrap_to_near(q_target, q_start)
        print(f"[{robot.name}] Moving to {name}...")
        trajectory = rtb.jtraj(q_start, q_target, steps)
        for q in trajectory.q:
            robot.q = q
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)
        print(f"✓ {name}")
        return q_target

    def animate_trajectory(self, robot, start_q, end_q, steps):
        q_path = rtb.jtraj(start_q, end_q, steps).q
        for q_config in q_path:
            robot.q = np.clip(q_config, robot.qlim[0, :], robot.qlim[1, :])
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)

    def print_pose(self, robot, label=""):
        print(f"\n{label}")
        q_deg = np.round(np.rad2deg(robot.q), 2)
        q_deg = q_deg.tolist()
        T = robot.fkine(robot.q)
        print(f"  Joints (deg): {q_deg}")
        print(f"  TCP Pos: {np.round(T.t, 3)}")