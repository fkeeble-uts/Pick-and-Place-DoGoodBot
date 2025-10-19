import numpy as np
import roboticstoolbox as rtb
import logging
from math import pi

class RobotController:
    def __init__(self, env, scene):
        self.env = env
        self.scene = scene

    def find_ikine(self, robot, target_tr, initial_q_guess=None, ignore_var="", ignore_rotation=False, hover_max=None):
        """
        Generalized IK function with a hover-zone check for x, y, or z axes.
        """
        num_attempts = 100
        min_limits = robot.qlim[0, :]
        max_limits = robot.qlim[1, :]
        mask = [1, 1, 1, 1, 1, 1]

        # Map the ignored variable to its index (0=x, 1=y, 2=z)
        coord_map = {"x": 0, "y": 1, "z": 2}
        if ignore_var in coord_map:
            mask[coord_map[ignore_var]] = 0
            
        if ignore_rotation:
            mask[3:6] = [0, 0, 0]

        for i in range(num_attempts):
            if i == 0 and initial_q_guess is not None:
                q_guess = np.deg2rad(initial_q_guess) if len(initial_q_guess) == len(robot.q) else robot.q
            else:
                q_guess = np.random.uniform(low=min_limits, high=max_limits)

            ik_result = robot.ikine_LM(target_tr, q0=q_guess, mask=mask)

            if ik_result.success:
                solution = ik_result.q
                if np.all((solution >= min_limits) & (solution <= max_limits)):
                    
                    # Check if this is a hover-find call for a specific axis
                    if hover_max is not None and ignore_var in coord_map:
                        axis_index = coord_map[ignore_var]
                        
                        # Perform an FK check on the potential solution
                        solution_pose = robot.fkine(solution)
                        solution_coord = solution_pose.t[axis_index]
                        target_coord = target_tr.t[axis_index]
                        
                        # Use min() and max() to handle both positive and negative hover_max
                        lower_bound = min(target_coord, target_coord + hover_max)
                        upper_bound = max(target_coord, target_coord + hover_max)
                        
                        # Check if the coordinate is in the valid range
                        if lower_bound <= solution_coord <= upper_bound:
                            print(f"IK solution found on attempt {i+1} with valid hover on axis '{ignore_var}'.")
                            return solution, True
                        # If not, this solution is invalid, so the loop continues
                    
                    else:
                        # This is a standard IK call, so return the solution immediately
                        print(f"IK solution found on attempt {i+1}.")
                        return solution, True

        logging.warning(f"Failed to find a valid IK solution after {num_attempts} attempts.")
        return robot.q, False


    def wrap_to_near(self, q_goal, q_ref):
        """Wrap joint angles to nearest equivalent to reference."""
        return q_ref + (q_goal - q_ref + pi) % (2 * pi) - pi

    def move_to_q(self, robot, q_target, steps=None, name="", carry_object=None):
        """Move robot to target joint angles, optionally carrying an object."""
        if steps is None:
            steps = self.scene.TRAJ_STEPS

        q_start = robot.q.copy()
        q_target = self.wrap_to_near(q_target, q_start)
        
        print(f"[{robot.name}] Moving to {name}...")
        trajectory = rtb.jtraj(q_start, q_target, steps)
        
        for q in trajectory.q:
            robot.q = q
            if carry_object is not None:
                T_tcp = robot.fkine(q)
                carry_object.T = T_tcp.A
            self.env.step(self.scene.SIM_STEP_TIME)
        
        print(f"✓ {name}")
        return q_target

    def print_pose(self, robot, label=""):
        """Print current pose info."""
        print(f"\n{label}")
        q_deg = np.round(np.rad2deg(robot.q), 2)
        T = robot.fkine(robot.q)
        print(f"  Joints (deg): {q_deg}")
        print(f"  TCP Pos: {np.round(T.t, 3)}")