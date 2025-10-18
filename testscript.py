import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid, Box
from math import pi

from Drinkbot import Drinkbot
from Ingredientbot import IngredientBot
from Glassbot import Glassbot

# ----------------------------------------------------
# I. CONSTANTS & CONFIGURATION
# ----------------------------------------------------

# --- Environment & Timing ---
SIM_STEP_TIME = 0.02 # Time step for swift.step()
TRAJ_STEPS = 60      # Default steps for joint trajectory movement

# --- Shared Object Dimensions ---
GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06
BUTTON_RADIUS = 0.05          # Radius of red button
BUTTON_HEIGHT = 0.03          # Height of red button
BUTTON_BASE_LENGTH = 0.12            # Base (cuboid) dimensions
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

# --- Global State Variable (used by R1's gripper) ---
_finger_gap_r1 = FINGER_GAP_OPEN

# ----------------------------------------------------
# II. SWIFT OBJECT PARAMETERS
# ----------------------------------------------------

wall_height = 2.5      # metres
wall_thickness = 0.05  # metres
floor_height = 0.01    # slightly raised to avoid flicker

# Table 1 (large workstation table)
table1_length = 4.0
table1_width  = 0.75
table1_height = 1.0
table1_offset_from_wall = 0.5
table1_center_y = -1.5 + wall_thickness + table1_offset_from_wall + table1_width / 2

# Table 2 (smaller / front table)
table2_length = 1.5
table2_width  = 0.7
table2_height = 1.0
table2_spacing = 1.0
table2_center_y = table1_center_y + (table1_width / 2) + table2_spacing + (table2_width / 2)

# Table 3 (drinks shelf)
table3_length = 4.0
table3_width  = 0.75
table3_height = 1.5
table3_offset_from_wall = 0
table3_center_y = -3 + wall_thickness + table1_offset_from_wall + table3_width / 2

# Glass table (for cups)
glass_table_length = 0.4
glass_table_width  = 0.7
glass_table_height = 1.05
glass_table_center_x = table1_length/2 + glass_table_length/2 + 0.1  # just left of Table 1
glass_table_center_y = table1_center_y

# Ingredients table (for ingredients - mirrored on opposite side)
ingredients_table_length = 0.4
ingredients_table_width  = 0.7
ingredients_table_height = 1.05
ingredients_table_center_x = -(table1_length/2 + ingredients_table_length/2 + 0.1)  # mirrored on right of Table 1
ingredients_table_center_y = table1_center_y

# Emergency stop button
button_center_x = 0.5  # along table length
button_center_y = table2_center_y + table2_width / 2 - 0.1  # near the front edge
button_center_z = table2_height + BUTTON_BASE_HEIGHT / 2 -0.01  # sits on table

# LED / glow parameters
base_color      = [0.1, 0.1, 0.15, 1]     # dark graphite
top_color       = [0.0, 0.6, 0.8, 1]      # neon cyan
button_base_color = [0.2, 0.2, 0.2, 1]       # Dark gray
button_color = [0.8, 0, 0, 1] 
top_glow_color  = [0.0, 0.8, 1.0, 0.3]    # semi-transparent glow
led_color       = [0.0, 0.8, 1.0, 0.6]    # bright cyan
led_height      = 0.05                     # LED thickness
led_offset      = 0.01                     # slightly above floor
led_margin      = 0.02                     # tiny extension beyond table edges
num_wraps       = 3                        # wrap-around LED rings
wrap_spacing_factor = 0.25                 # fraction of table height between wraps

# ----------------------------------------------------
# III. ROBOT BASE POSES
# ----------------------------------------------------

ROBOT_BASE_POSES = {
    "R1_ICE_GLASS": SE3(1.6, table1_center_y, table1_height + floor_height),
    "R2_ALCOHOL":   SE3(-1.6, table1_center_y, table1_height + floor_height),
    "R3_MIXERS":    SE3(0.0, table1_center_y, table1_height + floor_height),
    "R4_SERVER":    SE3(0.0, table2_center_y, table2_height + floor_height),
}

# ----------------------------------------------------
# IV. ENVIRONMENT SETUP (Swift Simulation)
# ----------------------------------------------------

env = swift.Swift()
env.launch(realtime=True)

# --- Floor ---
floor = Cuboid(scale=[6, 4, 0.02],
               color=[0.25, 0.3, 0.35, 1],
               pose=SE3(0, -0.5, floor_height))
env.add(floor)

# --- Walls ---
back_wall = Cuboid(scale=[6, wall_thickness, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(0, -2.5, wall_height/2))
env.add(back_wall)

left_wall = Cuboid(scale=[wall_thickness, 4, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(-3, -0.5, wall_height/2))
env.add(left_wall)

right_wall = Cuboid(scale=[wall_thickness, 4, wall_height],
                    color=[0.85, 0.85, 0.9, 1],
                    pose=SE3(3, -0.5, wall_height/2))
env.add(right_wall)

# ----------------------------------------------------
# V. TABLES
# ----------------------------------------------------

tables = [
    {
        "name": "Workstation",
        "length": table1_length,
        "width": table1_width,
        "height": table1_height,
        "center": SE3(0, table1_center_y, 0),
        "leds": False  # no LEDs on back table
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
        "leds": False  # no LEDs on back table
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

    # Base
    base = Cuboid(scale=[l, w, h-0.05],
                  color=base_color,
                  pose=SE3(cx, cy, (h-0.05)/2))
    env.add(base)

    # Top
    top = Cuboid(scale=[l, w, 0.05],
                 color=top_color,
                 pose=SE3(cx, cy, h - 0.025))
    env.add(top)

    # Glow (slightly larger for small tables only)
    glow_scale_x = l*1.05 if l <= table2_length else l
    glow_scale_y = w*1.05 if l <= table2_length else w
    glow = Cuboid(scale=[glow_scale_x, glow_scale_y, 0.02],
                  color=top_glow_color,
                  pose=SE3(cx, cy, h - 0.015))
    env.add(glow)

    # LED strips for small tables only
    if t["leds"]:
        # Bottom LED
        led = Cuboid(scale=[l + led_margin*2, w + led_margin*2, led_height],
                     color=led_color,
                     pose=SE3(cx, cy, (led_height/2)+led_offset))
        env.add(led)

        # Wrap-around LED rings
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

# ----------------------------------------------------
# V. ROBOT INSTANTIATION & PLACEMENT
# ----------------------------------------------------

# Robot 1: Glass & Ice Handler
robot1 = Glassbot()
robot1.base = ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)


# Robot 2: Alcohol Pourer
robot2 = Drinkbot()
robot2.q = robot2.home_q
robot2.base = ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

# Robot 3: Mixer Adder
robot3 = IngredientBot()
robot3.q = robot3.home_q
robot3.base = ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

# Robot 4: Server (placeholder)
# robot4 = Drinkbot4()
# robot4.base = ROBOT_BASE_POSES["R4_SERVER"]
# robot4.add_to_env(env)

# ----------------------------------------------------
# VI. OBJECT DEFINITIONS (Glass, Ice Dispenser, etc.)
# ----------------------------------------------------

# ----------------------------------------------------
# V. GLASSES ON Glass TABLE
# ----------------------------------------------------

# Glass parameters
glass_radius = 0.025  # radius of glass
glass_height = 0.1    # height of glass
glass_color = [1.0, 0.4, 0.0, 0.7]  # bright, slightly transparent

# Define proportional offsets along table width (y-axis)
width_fractions = [0.1, 0.5, 0.9]  # near back edge, middle, near front edge
# Define proportional offsets along table length (x-axis)
length_fractions = [0.1, 0.5, 0.9]  # left, middle, right

# Loop to create individual glasses
glass_objects = []

for yf in width_fractions:
    for xf in length_fractions:
        # Calculate absolute positions on table
        x_pos = glass_table_center_x - glass_table_length/2 + xf * glass_table_length
        y_pos = glass_table_center_y - glass_table_width/2 + yf * glass_table_width
        z_pos = glass_table_height + glass_height / 2  # standing on table

        # Create Cylinder standing upright
        glass = Cylinder(radius=glass_radius,
                         length=glass_height,
                         color=glass_color,
                         pose=SE3(x_pos, y_pos, z_pos))  # no rotation, upright along z
        env.add(glass)
        glass_objects.append(glass)

# --- Drinks on drink shelf ---
# Drink parameters
drink_radius = 0.05  # radius of drink
drink_height = 0.2    # height of drink
drink_color = [0, 0, 0.4, 0.7]  # bright, slightly transparent

drink_count = 9
drink_gaps = (2 - drink_radius) / drink_count + 0.03

for i in range(drink_count):
    # Create Cylinder standing upright
    drink = Cylinder(radius=drink_radius,
                     length=drink_height,
                     color=drink_color,
                     pose=SE3(1 - drink_gaps * i, table3_center_y, 
                              table3_height + drink_height/2))  # no rotation, upright along z
    env.add(drink)
    print("Added drink at location: ", 1 - drink_gaps * i, table3_center_y, table3_height + drink_height/2)

# ----------------------------------------------------
# VI. INGREDIENTS TABLE OBJECTS (Chopping Boards + Cubes)
# ----------------------------------------------------

# --- Adjustable Parameters ---

# Chopping board dimensions
board_length = 0.3
board_width  = 0.185
board_height = 0.02

# Cube (ingredient) dimensions
cube_size = 0.025
cube_spacing_x = 0.07
cube_spacing_y = 0.035
cube_height_offset = board_height/2 + cube_size/2   # sits just above board

# Colors
board_color = [1.0, 1.0, 1.0, 1.0]   # pure white boards
cube_colors = {
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green":  [0.0, 1.0, 0.0, 1.0],
    "blue":   [0.0, 0.0, 1.0, 1.0],
}

# Positioning offsets (fractions of table surface)
# Define 3 boards across the table width (front, middle, back)
board_fractions_y = [0.2, 0.5, 0.8]
board_center_x_fraction = 0.5  # centered along table length

ingredient_objects = []
cube_objects = []

for i, yf in enumerate(board_fractions_y):
    # Compute chopping board world position
    x_pos = (ingredients_table_center_x - ingredients_table_length/2
             + board_center_x_fraction * ingredients_table_length)
    y_pos = (ingredients_table_center_y - ingredients_table_width/2
             + yf * ingredients_table_width)
    z_pos = ingredients_table_height + board_height / 2  # sits on table

    # Create chopping board (Cuboid)
    board = Cuboid(
        scale=[board_length, board_width, board_height],
        color=board_color,
        pose=SE3(x_pos, y_pos, z_pos)
    )
    env.add(board)
    ingredient_objects.append(board)

    # 3x3 grid of cubes on each board
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

    print(f"Added 3x3 grid of {color_key} cubes on board {i+1} at ({x_pos:.2f}, {y_pos:.2f})")

print(f"Total cubes: {len(cube_objects)}")

# --- 3. Alcohol Bottle (PLACEHOLDER for R2) ---
# ALCOHOL_POSE = SE3(x, y, z) 
# alcohol_bottle = Cuboid(scale=[...], pose=ALCOHOL_POSE, color=[0.7, 0.1, 0.1, 1.0])
# env.add(alcohol_bottle)


# --- 4. Mixer Bottles (PLACEHOLDER for R3) ---
# MIXER_POSES = [...]
# mixer_bottle_1 = ...
# env.add(mixer_bottle_1)


# --- 5. Serving Area/Dropoff (PLACEHOLDER for R4) ---
# SERVE_AREA = Cuboid(scale=[...], pose=SE3(...), color=[0.5, 0.5, 0.5, 0.5])
# env.add(SERVE_AREA)

# ============================================================================
# VIII. MOVEMENT HELPERS
# ============================================================================

def wrap_to_near(q_goal, q_ref):
    """Wrap joint angles to nearest equivalent to reference."""
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi

def move_to_q(robot, q_target, steps=TRAJ_STEPS, name="", carry_object=None):
    """Move robot to target joint angles, optionally carrying an object."""
    q_start = robot.q.copy()
    q_target = wrap_to_near(q_target, q_start)
    
    print(f"[{robot.name}] Moving to {name}...")
    trajectory = rtb.jtraj(q_start, q_target, steps)
    
    for q in trajectory.q:
        robot.q = q
        
        if carry_object is not None:
            T_tcp = robot.fkine(q)
            carry_object.T = T_tcp.A
        
        env.step(SIM_STEP_TIME)
    
    print(f"✓ {name}")
    return q_target

def print_pose(robot, label=""):
    """Print current pose info."""
    print(f"\n{label}")
    q_deg = np.round(np.rad2deg(robot.q), 2)
    T = robot.fkine(robot.q)
    print(f"  Joints (deg): {q_deg}")
    print(f"  TCP Pos: {np.round(T.t, 3)}")

# ============================================================================
# IX. SAVED JOINT POSES FROM TEACH MODE
# ============================================================================

# R1 (Glassbot) - Collected from teach mode
R1_POSES = {
    "HOME": np.deg2rad(np.array([0., 0., 0., 0., 0., 0.])),
    "GLASS_APPROACH": np.deg2rad(np.array([0., 40.68, 16.82, 9.09, 0., 0.])),
    "GLASS_PICKUP": np.deg2rad(np.array([0., 62.27, 0.45, 10.91, 0., 0.])),
    "LIFT_CLEAR": np.deg2rad(np.array([0., 39.55, -11.82, 10.91, 0., 0.])),
    "ICE_MACHINE": np.deg2rad(np.array([-87.95, 52.05, 3.52, 10.91, 0., 0.])),
    "HANDOFF": np.deg2rad(np.array([-180., 92.95, -61.93, 10.91, 0., 0.])),
}

# R2 (Drinkbot) - Will need to collect these with teach mode later
R2_POSES = {
    "HOME": robot2.q.copy(),
    "HANDOFF_PICKUP": robot2.q.copy(),  # Position to receive glass
    "HANDOFF_PLACE": robot2.q.copy(),    # Position to place glass back
}

# ============================================================================
# X. GLASS OBJECT TRACKING
# ============================================================================

glass_index = 4  # Use middle glass
target_glass = glass_objects[glass_index]
held_by_r1 = False
held_by_r2 = False

# ============================================================================
# XI. ROBOT 1 SEQUENCE - PICK UP GLASS
# ============================================================================

print("\n" + "="*70)
print(">>> ROBOT 1: PICKING UP GLASS <<<")
print("="*70 + "\n")

q_now_r1 = R1_POSES["HOME"]
robot1.q = q_now_r1
print_pose(robot1, "R1 at HOME")
time.sleep(0.5)

# Step 1: Approach glass
print("\n[R1] Approaching glass...")
q_now_r1 = move_to_q(robot1, R1_POSES["GLASS_APPROACH"], steps=50, name="Glass Approach")
print_pose(robot1, "R1 at GLASS_APPROACH")
time.sleep(0.5)

# Step 2: Move to glass level
print("\n[R1] Moving to glass pickup position...")
q_now_r1 = move_to_q(robot1, R1_POSES["GLASS_PICKUP"], steps=40, name="Glass Pickup", 
                      carry_object=target_glass)
print_pose(robot1, "R1 at GLASS_PICKUP")
time.sleep(0.5)

# Step 3: Simulate gripper closing
print("\n[R1] Closing gripper...")
held_by_r1 = True
time.sleep(0.5)

# Step 4: Lift glass
print("\n[R1] Lifting glass...")
q_now_r1 = move_to_q(robot1, R1_POSES["LIFT_CLEAR"], steps=40, name="Lift Clear",
                      carry_object=target_glass)
print_pose(robot1, "R1 at LIFT_CLEAR")
time.sleep(0.5)

# Step 5: Move to ice machine
print("\n[R1] Moving to ice machine...")
q_now_r1 = move_to_q(robot1, R1_POSES["ICE_MACHINE"], steps=60, name="Ice Machine",
                      carry_object=target_glass)
print_pose(robot1, "R1 at ICE_MACHINE")
print("[R1] Simulating ice fill...")
time.sleep(1.0)

# Step 6: Move to handoff location
print("\n[R1] Moving to handoff location...")
q_now_r1 = move_to_q(robot1, R1_POSES["HANDOFF"], steps=60, name="Handoff",
                      carry_object=target_glass)
print_pose(robot1, "R1 at HANDOFF")
time.sleep(0.5)

# ============================================================================
# XII. ROBOT 2 SEQUENCE - RECEIVE GLASS FROM R1
# ============================================================================

print("\n" + "="*70)
print(">>> ROBOT 2: RECEIVING GLASS FROM R1 <<<")
print("="*70 + "\n")

q_now_r2 = R2_POSES["HOME"]
robot2.q = q_now_r2
print_pose(robot2, "R2 at HOME")
time.sleep(0.5)

# Step 1: Move to handoff pickup position
print("\n[R2] Moving to handoff location to receive glass...")
# Use IK to reach the handoff position (same world location as R1's TCP)
T_handoff = robot1.fkine(R1_POSES["HANDOFF"])
print(f"  Handoff position (world): {np.round(T_handoff.t, 3)}")

# Try to solve IK for R2 to reach handoff position using current R2 pose as guess
sol = robot2.ikine_LM(T_handoff, q0=q_now_r2, mask=[1,1,1,0,0,0], joint_limits=True)
if sol.success:
    q_now_r2 = move_to_q(robot2, sol.q, steps=60, name="Handoff Pickup")
    print_pose(robot2, "R2 at HANDOFF")
    time.sleep(0.5)
    
    # Step 2: Simulate gripper opening to receive glass
    print("\n[R2] Opening gripper to receive glass from R1...")
    time.sleep(0.5)
    
    # Step 3: Simulate gripper closing around glass
    print("\n[R2] Gripper closing around glass...")
    held_by_r1 = False
    held_by_r2 = True
    time.sleep(0.5)
    
    # Step 4: Move glass to R2's TCP
    print("\n[R2] Lifting glass...")
    T_tcp_r2 = robot2.fkine(q_now_r2)
    target_glass.T = T_tcp_r2.A
    
    # Step 5: Return to home
    print("\n[R2] Returning to home...")
    q_now_r2 = move_to_q(robot2, R2_POSES["HOME"], steps=60, name="Home",
                         carry_object=target_glass)
    print_pose(robot2, "R2 at HOME with glass")
    time.sleep(0.5)
    
else:
    print("[R2] ❌ IK failed - could not reach handoff position")

# ============================================================================
# XIII. SUMMARY
# ============================================================================

print("\n" + "="*70)
print(">>> HANDOFF COMPLETE <<<")
print("="*70)
print(f"\nGlass transfer status:")
print(f"  Held by R1: {held_by_r1}")
print(f"  Held by R2: {held_by_r2}")
print(f"  Glass position: {np.round(target_glass.T[0:3, 3], 3)}")

time.sleep(1.0)


# --- STEP 4: R2 Picks Up Glass and Pours Alcohol (PLACEHOLDER) ---
# print("--- R2: Pouring Alcohol ---")
# q_now_r2 = move_joint_traj(robot2, q_now_r2, q_approach_h1_r2) # Move R2 to Handover 1
# gripper_close(robot2, move_mesh=True, steps=30, mesh_to_grip=wine_glass) # R2 picks up glass
# q_now_r2 = move_joint_traj(robot2, q_now_r2, ALCOHOL_FILL_POSE_R2, carry_mesh=wine_glass) # Move to alcohol station
# # *** INSERT POURING LOGIC HERE ***
# q_now_r2 = move_joint_traj(robot2, q_now_r2, q_approach_h2_r2, carry_mesh=wine_glass) # Move to Handover 2
# # ... place and release glass at Handover 2 (for R3) ...
# q_now_r2 = move_joint_traj(robot2, q_now_r2, q_home_r2)


# --- STEP 5: R3 Picks Up Glass and Adds Mixers (PLACEHOLDER) ---
# print("--- R3: Adding Mixers ---")
# # ... R3 picks up glass from Handover 2 ...
# q_now_r3 = move_joint_traj(robot3, q_now_r3, MIXER_FILL_POSE_R3, carry_mesh=wine_glass) # Move to mixer station
# # *** INSERT MIXING LOGIC HERE ***
# q_now_r3 = move_joint_traj(robot3, q_now_r3, q_approach_h3_r3, carry_mesh=wine_glass) # Move to Handover 3
# # ... place and release glass at Handover 3 (for R4) ...
# q_now_r3 = move_joint_traj(robot3, q_now_r3, q_home_r3)


# --- STEP 6: R4 Serves Drink (PLACEHOLDER) ---
# print("--- R4: Serving Customer ---")
# # ... R4 picks up glass from Handover 3 ...
# q_now_r4 = move_joint_traj(robot4, q_now_r4, SERVE_POSE_R4, carry_mesh=wine_glass) # Move to serving location
# # ... R4 places glass at customer area ...
# q_now_r4 = move_joint_traj(robot4, q_now_r4, q_home_r4)


# ----------------------------------------------------
# XI. EXECUTION & HOLD
# ----------------------------------------------------

env.hold()

