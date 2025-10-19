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
from Serverbot import Serverbot

# Log config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# --- Environment & Timing ---
SIM_STEP_TIME = 0.02
TRAJ_STEPS = 60

# --- Shared Object Dimensions ---
GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06
BUTTON_RADIUS = 0.05
BUTTON_HEIGHT = 0.03
BUTTON_BASE_LENGTH = 0.12
BUTTON_BASE_WIDTH = 0.12
BUTTON_BASE_HEIGHT = 0.02

# --- Gripper Visual Dimensions ---
FINGER_LENGTH = 0.08
FINGER_THICKNESS = 0.01
FINGER_HEIGHT = 0.08
FINGER_GAP_OPEN = 0.06
FINGER_GAP_CLOSED = 0.055
FINGER_BACK_OFFSET = 0.02
FINGER_Z_OFFSET = -0.03

# --- Swift Object Parameters ---
wall_height = 2.5
wall_thickness = 0.05
floor_height = 0.01

# Table 1 (large workstation table)
table1_length = 4.0
table1_width = 0.75
table1_height = 1.0
table1_offset_from_wall = 0.5
table1_center_y = -1.5 + wall_thickness + table1_offset_from_wall + table1_width / 2

# Table 2 (smaller / front table)
table2_length = 1.5
table2_width = 0.7
table2_height = 1.0
table2_spacing = 1.0
table2_center_y = table1_center_y + (table1_width / 2) + table2_spacing + (table2_width / 2)

# Table 3 (drinks shelf)
table3_length = 4.0
table3_width = 0.325
table3_height = 1.2
table3_center_y = -1.75 + wall_thickness + table3_width / 2

# Glass table (for cups)
glass_table_length = 0.4
glass_table_width = 0.7
glass_table_height = 1.05
glass_table_center_x = table1_length/2 + glass_table_length/2 + 0.1
glass_table_center_y = table1_center_y

# Ingredients table
ingredients_table_length = 0.4
ingredients_table_width = 0.7
ingredients_table_height = 1.05
ingredients_table_center_x = -(table1_length/2 + ingredients_table_length/2 + 0.1)
ingredients_table_center_y = table1_center_y

# Emergency stop button
button_center_x = 0.5
button_center_y = table2_center_y + table2_width / 2 - 0.1
button_center_z = table2_height + BUTTON_BASE_HEIGHT / 2 - 0.01

# LED / glow parameters
base_color = [0.1, 0.1, 0.15, 1]
top_color = [0.0, 0.6, 0.8, 1]
button_base_color = [0.2, 0.2, 0.2, 1]
button_color = [0.8, 0, 0, 1]
top_glow_color = [0.0, 0.8, 1.0, 0.3]
led_color = [0.0, 0.8, 1.0, 0.6]
led_height = 0.05
led_offset = 0.01
led_margin = 0.02
num_wraps = 3
wrap_spacing_factor = 0.25

# --- Robot Base Poses ---
ROBOT_BASE_POSES = {
    "R1_ICE_GLASS": SE3(1.6, table1_center_y, table1_height + floor_height),
    "R2_ALCOHOL": SE3(0.0, table1_center_y, table1_height + floor_height),
    "R3_MIXERS": SE3(-1.6, table1_center_y, table1_height + floor_height),
    "R4_SERVER": SE3(0.2, table2_center_y, table2_height + floor_height),
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

env = swift.Swift()
env.launch(realtime=True)

# --- Floor ---
floor = Cuboid(scale=[6, 3.25, 0.02],
               color=[0.25, 0.3, 0.35, 1],
               pose=SE3(0, -0.125, floor_height))
env.add(floor)

# --- Walls ---
back_wall = Cuboid(scale=[6, wall_thickness, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(0, -1.75, wall_height/2))
env.add(back_wall)

left_wall = Cuboid(scale=[wall_thickness, 3.25, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(-3, -0.125, wall_height/2))
env.add(left_wall)

right_wall = Cuboid(scale=[wall_thickness, 3.25, wall_height],
                    color=[0.85, 0.85, 0.9, 1],
                    pose=SE3(3, -0.125, wall_height/2))
env.add(right_wall)

# --- Tables ---
tables = [
    {
        "name": "Workstation",
        "length": table1_length,
        "width": table1_width,
        "height": table1_height,
        "center": SE3(0, table1_center_y, 0),
        "leds": False
    },
    {
        "name": "UR3e Table",
        "length": table2_length,
        "width": table2_width,
        "height": table2_height,
        "center": SE3(0, table2_center_y, 0),
        "leds": True
    },
    {
        "name": "Glass Table",
        "length": glass_table_length,
        "width": glass_table_width,
        "height": glass_table_height,
        "center": SE3(glass_table_center_x, glass_table_center_y, 0),
        "leds": True
    },
    {
        "name": "Drinks Shelf",
        "length": table3_length,
        "width": table3_width,
        "height": table3_height,
        "center": SE3(0, table3_center_y, 0),
        "leds": False
    },
    {
        "name": "Ingredients Table",
        "length": ingredients_table_length,
        "width": ingredients_table_width,
        "height": ingredients_table_height,
        "center": SE3(ingredients_table_center_x, ingredients_table_center_y, 0),
        "leds": True  
    }
]

for t in tables:
    cx, cy, cz = t["center"].t
    h = t["height"]
    l = t["length"]
    w = t["width"]

    base = Cuboid(scale=[l, w, h-0.05],
                  color=base_color,
                  pose=SE3(cx, cy, (h-0.05)/2))
    env.add(base)

    top = Cuboid(scale=[l, w, 0.05],
                 color=top_color,
                 pose=SE3(cx, cy, h - 0.025))
    env.add(top)

    glow_scale_x = l*1.05 if l <= table2_length else l
    glow_scale_y = w*1.05 if l <= table2_length else w
    glow = Cuboid(scale=[glow_scale_x, glow_scale_y, 0.02],
                  color=top_glow_color,
                  pose=SE3(cx, cy, h - 0.015))
    env.add(glow)

    if t["leds"]:
        led = Cuboid(scale=[l + led_margin*2, w + led_margin*2, led_height],
                     color=led_color,
                     pose=SE3(cx, cy, (led_height/2)+led_offset))
        env.add(led)

        for i in range(1, num_wraps+1):
            wrap_z = led_height + (h - 0.05 - led_height) * wrap_spacing_factor * i
            wrap_ring = Cuboid(scale=[l + led_margin*2, w + led_margin*2, led_height],
                               color=led_color,
                               pose=SE3(cx, cy, wrap_z))
            env.add(wrap_ring)

# --- Emergency stop button ---
stop_base = Cuboid(
    scale=[BUTTON_BASE_LENGTH, BUTTON_BASE_WIDTH, BUTTON_BASE_HEIGHT],
    color=button_base_color,
    pose=SE3(button_center_x, button_center_y, 
             button_center_z + BUTTON_BASE_HEIGHT/2)
)
env.add(stop_base)

red_button = Cylinder(
    radius=BUTTON_RADIUS,
    length=BUTTON_HEIGHT,
    color=button_color,
    pose=SE3(button_center_x, button_center_y, 
             button_center_z + BUTTON_BASE_HEIGHT/2 + BUTTON_HEIGHT/2)
)
env.add(red_button)

# --- Glasses on Glass Table ---
glass_radius = 0.025
glass_height = 0.1
glass_color = [1.0, 0.4, 0.0, 0.7]

width_fractions = [0.1, 0.5, 0.9]
length_fractions = [0.1, 0.5, 0.9]

glass_objects = []

for yf in width_fractions:
    for xf in length_fractions:
        x_pos = glass_table_center_x - glass_table_length/2 + xf * glass_table_length
        y_pos = glass_table_center_y - glass_table_width/2 + yf * glass_table_width
        z_pos = glass_table_height + glass_height / 2

        glass = Cylinder(radius=glass_radius,
                         length=glass_height,
                         color=glass_color,
                         pose=SE3(x_pos, y_pos, z_pos))
        env.add(glass)
        glass_objects.append(glass)

# --- Drinks on drink shelf ---
drink_radius = 0.05
drink_height = 0.2
drink_color = [0, 0, 0.4, 0.7]

drink_count = 9
drink_gaps = (2 - drink_radius) / drink_count + 0.03
drink_poses = []

for i in range(drink_count):
    drink = Cylinder(radius=drink_radius,
                     length=drink_height,
                     color=drink_color,
                     pose=SE3(1 - drink_gaps * i, table3_center_y, 
                              table3_height + drink_height/2))
    env.add(drink)
    drink_poses.append(SE3(1 - drink_gaps * i, table3_center_y, table3_height + drink_height/2))

# --- Ingredients Table Objects ---
board_length = 0.3
board_width = 0.185
board_height = 0.02

cube_size = 0.025
cube_spacing_x = 0.07
cube_spacing_y = 0.035
cube_height_offset = board_height/2 + cube_size/2

board_color = [1.0, 1.0, 1.0, 1.0]
cube_colors = {
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
}

board_fractions_y = [0.2, 0.5, 0.8]
board_center_x_fraction = 0.5

ingredient_objects = []
cube_objects = []

for i, yf in enumerate(board_fractions_y):
    x_pos = (ingredients_table_center_x - ingredients_table_length/2
             + board_center_x_fraction * ingredients_table_length)
    y_pos = (ingredients_table_center_y - ingredients_table_width/2
             + yf * ingredients_table_width)
    z_pos = ingredients_table_height + board_height / 2

    board = Cuboid(
        scale=[board_length, board_width, board_height],
        color=board_color,
        pose=SE3(x_pos, y_pos, z_pos)
    )
    env.add(board)
    ingredient_objects.append(board)

    color_key = list(cube_colors.keys())[i]
    cube_color = cube_colors[color_key]

    total_grid_size_x = 3 * cube_size + 2 * cube_spacing_x
    total_grid_size_y = 3 * cube_size + 2 * cube_spacing_y
    x_start = x_pos - total_grid_size_x / 2 + cube_size / 2
    y_start = y_pos - total_grid_size_y / 2 + cube_size / 2

    for row in range(3):
        for col in range(3):
            cube_x = x_start + col * (cube_size + cube_spacing_x)
            cube_y = y_start + row * (cube_size + cube_spacing_y)
            cube_z = z_pos + cube_height_offset

            cube = Cuboid(
                scale=[cube_size, cube_size, cube_size],
                color=cube_color,
                pose=SE3(cube_x, cube_y, cube_z)
            )
            env.add(cube)
            ingredient_objects.append(cube)
            cube_objects.append(cube)

# ============================================================================
# ROBOT INSTANTIATION & PLACEMENT
# ============================================================================

# Robot 1: Glass & Ice Handler
robot1 = GlassBot()
robot1.base = ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

# Robot 2: Alcohol Pourer
robot2 = DrinkBot()
robot2.q = robot2.home_q
robot2.base = ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

# Robot 3: Mixer Adder
robot3 = IngredientBot()
robot3.q = robot3.home_q
robot3.base = ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

# Robot 4: Server
robot4 = Serverbot()
robot4.base = ROBOT_BASE_POSES["R4_SERVER"] * SE3.Rx(pi/2) * SE3.Ry(pi/2)
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
        target_pose = drink_poses[3] @ SE3.Rx(pi/2)
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
                    env.step(SIM_STEP_TIME) # Update the simulation
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