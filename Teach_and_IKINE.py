import swift
import roboticstoolbox as rtb
import numpy as np
import logging
from math import pi
import time
from spatialmath import SE3
from spatialgeometry import Cylinder, Cuboid, Box

# --- Import all robot classes ---
from IngredientBot import IngredientBot
from DrinkBot import DrinkBot
from GlassBot import GlassBot
from ServerBot import ServerBot
from EnvironmentSetup import Scene

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([0, 3, 4], [0, 0, 0.5])
scene = Scene(env)

# ============================================================================
# ROBOT INSTANTIATION & PLACEMENT
# ============================================================================

# Robot 1: Glass & Ice Handler
robot1 = GlassBot()
robot1.base = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

# Robot 2: Alcohol Pourer
robot2 = DrinkBot()
robot2.q = robot2.home_q
robot2.base = scene.ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

# Robot 3: Mixer Adder
robot3 = IngredientBot()
robot3.q = robot3.home_q
robot3.base = scene.ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

# Robot 4: Server
robot4 = ServerBot()
robot4.base = scene.ROBOT_BASE_POSES["R4_SERVER"] * SE3.Rx(pi/2) * SE3.Ry(pi/2)
robot4.add_to_env(env)

# ============================================================================
# MAIN TEACHING INTERFACE
# ============================================================================

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_ikine(robot, target_tr, initial_q_guess=None, ignore_var="", ignore_rotation=False, hover_max=None):
    """
    Generalized IK function with a robust hover-zone check for x, y, or z axes.
    """
    num_attempts = 300
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
                
                # --- NEW GENERALIZED HOVER VALIDATION BLOCK ---
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

def animate_trajectory(robot, sim_env, start_q, end_q, steps):
    """Generic trajectory animation function."""
    q_path = rtb.jtraj(start_q, end_q, steps).q
    for q_config in q_path:
        robot.q = np.clip(q_config, robot.qlim[0, :], robot.qlim[1, :])
        sim_env.step(0.02)

def slider_callback(value, joint_index, robot):
    """Generic callback for sliders."""
    new_q = robot.q.copy()
    new_q[joint_index] = np.deg2rad(value)
    robot.q = new_q
    
    q_degrees = np.round(np.rad2deg(robot.q), 2)
    print(f"Joint state (deg): {q_degrees}")
    
    tr_matrix = robot.fkine(robot.q) 
    print(f"End-effector pose:\n{np.round(tr_matrix.A, 4)}")

def create_sliders(robot, sim_env):
    """Creates and adds sliders to the environment for a given robot."""
    sliders = []
    for i in range(robot.n):
        slider = swift.Slider(
            cb=lambda value, j=i: slider_callback(value, j, robot),
            min=np.rad2deg(robot.qlim[0, i]),
            max=np.rad2deg(robot.qlim[1, i]),
            step=1,
            value=np.rad2deg(robot.q[i]),
            desc=f'Joint {i+1} Angle',
            unit='°'
        )
        sliders.append(slider)
    
    for s in sliders:
        sim_env.add(s)

if __name__ == "__main__":
    # --- CONTROL SWITCHES ---
    ROBOT_TO_LOAD = "IngredientBot"  # Options: "IngredientBot", "DrinkBot", "GlassBot", "ServerBot"
    RUN_IKINE = False  # False for sliders, True for IK test

    # --- ROBOT SELECTION ---
    if ROBOT_TO_LOAD == "DrinkBot":
        robot_arm = robot2
    elif ROBOT_TO_LOAD == "IngredientBot":
        robot_arm = robot3
    elif ROBOT_TO_LOAD == "GlassBot":
        robot_arm = robot1
    elif ROBOT_TO_LOAD == "ServerBot":
        robot_arm = robot4
    else:
        raise ValueError(f"Robot '{ROBOT_TO_LOAD}' is not a valid choice.")

    # --- MODE SELECTION ---
    if RUN_IKINE:
        print(f"Running IKINE for {ROBOT_TO_LOAD}...")

        hover_max = 0.5
        target_pose = scene.drink_poses[3] @ SE3.Rx(pi/2)
        print(f"Target Pose:\n{np.round(target_pose.A, 4)}\n")

        initial_q = robot_arm.q.copy()
        q_guess = [-74.207, 141.295, -31.751, 9.875, 103.964, -24.255]
        target_q, success = find_ikine(robot_arm, target_pose, ignore_var="y", ignore_rotation=False, hover_max=hover_max)

        if success:
            print("Animating robot to hover pose...")
            animate_trajectory(robot_arm, env, initial_q, target_q, 100)
            
            final_q_deg = np.round(np.rad2deg(robot_arm.q), 3)
            final_pose = robot_arm.fkine(robot_arm.q)

            print("\n--- IK Hover Solution Complete ---")
            print(f"Hover Joint State (deg): {final_q_deg}")
            print(f"Hover Robot Pose:\n{np.round(final_pose.A, 4)}")

            print("Following Cartesian path...")

            # generate a cartesian path
            start_pose = robot_arm.fkine(robot_arm.q)
            cartesian_path = rtb.ctraj(start_pose, target_pose, 50) # 50 steps

            for next_pose in cartesian_path:
                # Use the robot's current joint state as the initial guess for the next step.
                # This makes the solver very fast and stable.
                q_step, solved = find_ikine(robot_arm, next_pose, 
                                            initial_q_guess=np.rad2deg(robot_arm.q), 
                                            ignore_rotation=False)
                
                if solved:
                    # If a solution was found, update the robot's joints
                    robot_arm.q = q_step
                    env.step(scene.SIM_STEP_TIME) # Update the simulation
                else:
                    # If any step fails, stop the movement
                    logging.warning("IK failed for a step in the Cartesian path. Halting motion.")
                    break
        else:
            print("\n--- IK Hover Solution Failed ---")
        

    else:
        print(f"Running TEACH mode for {ROBOT_TO_LOAD}...")
        create_sliders(robot_arm, env)

    # --- Keep simulation open ---
    print("\nPress Ctrl+C in the terminal to exit.")
    while True:
        try:
            env.step(0.02)
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
            
    env.close()
    print("Program finished.")