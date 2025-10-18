import swift
import roboticstoolbox as rtb
import numpy as np
import logging
from math import pi
import time
from spatialmath import SE3

# --- Import all of your robot classes ---
from IngredientBot import IngredientBot
from Drinkbot import Drinkbot
from Glassbot import Glassbot
from Serverbot import Serverbot

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# --------------------------------------------------------------------
# GENERIC HELPER FUNCTIONS (Work with any rtb.DHRobot)
# --------------------------------------------------------------------

def find_ikine(robot, target_tr, initial_q_guess=None, ignore_rotation=False):
    """
    Generic IK function that works on any DHRobot object.
    """
    num_attempts = 25
    min_limits = robot.qlim[0, :]
    max_limits = robot.qlim[1, :]
    mask = [1, 1, 1, 0, 0, 0] if ignore_rotation else None

    for i in range(num_attempts):
        if i == 0 and initial_q_guess is not None:
            q_guess = np.deg2rad(initial_q_guess) if len(initial_q_guess) == len(robot.q) else robot.q
        else:
            q_guess = np.random.uniform(low=min_limits, high=max_limits)

        # Use the robot object's ikine_LM method
        ik_result = robot.ikine_LM(target_tr, q0=q_guess, mask=mask)

        if ik_result.success:
            solution = ik_result.q
            # Check if the solution is within joint limits
            if np.all((solution >= min_limits) & (solution <= max_limits)):
                print(f"IK solution found on attempt {i+1}.")
                return solution, True
    
    logging.warning(
        f"Failed to find a valid IK solution after {num_attempts} attempts."
    )
    return robot.q, False

def animate_trajectory(robot, sim_env, start_q, end_q, steps):
    """
    Generic trajectory animation function.
    """
    q_path = rtb.jtraj(start_q, end_q, steps).q
    for q_config in q_path:
        robot.q = np.clip(q_config, robot.qlim[0, :], robot.qlim[1, :])
        sim_env.step(0.02)

def slider_callback(value, joint_index, robot):
    """
    Generic callback for sliders. Needs the robot object passed to it.
    """
    new_q = robot.q.copy()
    new_q[joint_index] = np.deg2rad(value)
    robot.q = new_q
    
    q_degrees = np.round(np.rad2deg(robot.q), 2)
    print(f"Joint state (deg): {q_degrees}")
    
    tr_matrix = robot.fkine(robot.q) 
    print(f"End-effector pose:\n{np.round(tr_matrix.A, 4)}")

def create_sliders(robot, sim_env):
    """
    Creates and adds sliders to the environment for a given robot.
    """
    sliders = []
    for i in range(robot.n):
        slider = swift.Slider(
            # Use a lambda function to pass the robot object to the callback
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


# --------------------------------------------------------------------
# MAIN SIMULATION
# --------------------------------------------------------------------

if __name__ == "__main__":
    # --- CONTROL SWITCHES ---
    # Choose which robot to load: "IngredientBot", "Drinkbot", "Glassbot", "Serverbot"
    ROBOT_TO_LOAD = "Drinkbot"
    
    # Choose mode: False for sliders, True for IK test
    RUN_IKINE = False
    
    # --- SHARED SETUP ---
    env = swift.Swift()
    env.launch(realtime=True)
    
    # --- ROBOT SELECTION ---
    if ROBOT_TO_LOAD == "IngredientBot":
        robot_arm = IngredientBot()
    elif ROBOT_TO_LOAD == "Drinkbot":
        robot_arm = Drinkbot()
    elif ROBOT_TO_LOAD == "Glassbot":
        robot_arm = Glassbot()
    elif ROBOT_TO_LOAD == "Serverbot":
        robot_arm = Serverbot()
    else:
        raise ValueError(f"Robot '{ROBOT_TO_LOAD}' is not a valid choice.")
        
    robot_arm.add_to_env(env)
    
    # --- MODE SELECTION ---
    if RUN_IKINE:
        print(f"Running IKINE test for {ROBOT_TO_LOAD}...")

        # 1. Define a target pose
        target_pose = SE3(0.26, -0.9625, 0.29)
        print(f"Target Pose:\n{np.round(target_pose.A, 4)}\n")

        # 2. Run inverse kinematics using the generic function
        initial_q = robot_arm.q.copy()
        q_guess = [-74.207, 141.295, -31.751, 9.875, 103.964, -24.255]
        target_q, success = find_ikine(robot_arm, target_pose, initial_q_guess=q_guess, ignore_rotation=True)

        if success:
            # 3. Animate the robot to the target pose
            print("Animating robot...")
            animate_trajectory(robot_arm, env, initial_q, target_q, 100)
            
            # 4. Print the final robot pose and joint state
            final_q_deg = np.round(np.rad2deg(robot_arm.q), 3)
            final_pose = robot_arm.fkine(robot_arm.q)

            print("\n--- IK Test Complete ---")
            print(f"Final Joint State (deg): {final_q_deg}")
            print(f"Final Robot Pose:\n{np.round(final_pose.A, 4)}")
        else:
            print("\n--- IK Test Failed ---")

    else:
        # This is the "teach mode"
        print(f"Running TEACH mode for {ROBOT_TO_LOAD}...")
        create_sliders(robot_arm, env)

    # --- Keep simulation open ---
    print("\nPress Ctrl+C in the terminal to exit.")
    while True:
        try:
            # For IKINE mode, this just keeps the window open.
            # For SLIDER mode, this is the essential update loop.
            env.step(0.02)
            time.sleep(0.02)
        except KeyboardInterrupt:
            break
            
    env.close()
    print("Program finished.")