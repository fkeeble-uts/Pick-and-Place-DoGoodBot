import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid, Box

from Drinkbot import Drinkbot
from Ingredientbot import IngredientBot 
from Glassbot import Glassbot
# ----------------------------------------------------
# I. CONSTANTS & CONFIGURATION
# ----------------------------------------------------

# --- Environment & Timing ---
SIM_STEP_TIME = 0.02 # Time step for swift.step()
TRAJ_STEPS = 60      # Default steps for joint trajectory movement

# --- Robot Placement (Base Frames for all 4 robots) ---
# NOTE: You will need to determine the optimal placement for collision avoidance
ROBOT_BASE_POSES = {
    "R1_ICE_GLASS": SE3(0.0, 0.0, 0.0), # Current robot is placed at origin
    "R2_ALCOHOL": SE3(0.0, 1.0, 0.0),   # PLACEHOLDER: Define a clear pose for Robot 2
    "R3_MIXERS": SE3(-1.0, 0.0, 0.0),   # PLACEHOLDER: Define a clear pose for Robot 3
    "R4_SERVER": SE3(0.0, -1.0, 0.0),   # PLACEHOLDER: Define a clear pose for Robot 4
}

# --- Shared Object Dimensions ---
GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06

# --- Gripper Visual Dimensions ---
FINGER_LENGTH = 0.08
FINGER_THICKNESS = 0.01
FINGER_HEIGHT = 0.08
FINGER_GAP_OPEN = 0.06
FINGER_GAP_CLOSED = 0.055
FINGER_BACK_OFFSET = 0.02
FINGER_Z_OFFSET = -0.03

# --- Global State Variable (used by R1's gripper, will need to be made per-robot) ---
_finger_gap_r1 = FINGER_GAP_OPEN


# ----------------------------------------------------
# II. ENVIRONMENT SETUP (Swift Simulation)
# ----------------------------------------------------

# Initialize the Swift environment
env = swift.Swift()
env.launch(realtime=True)


# ----------------------------------------------------
# III. ROBOT INSTANTIATION & PLACEMENT
# ----------------------------------------------------

# --- Robot 1: Glass & Ice Handler ---
robot1 = Glassbot()
# Set the base transform for R1 (if not at origin, use SE3.T)
robot1.base = ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

# --- Robot 2: Alcohol Pourer ---
robot2 = Drinkbot()
robot2.base = ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

# --- Robot 3: Mixer Adder ---
robot3 = IngredientBot()
robot3.base = ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

# --- Robot 4: Server (PLACEHOLDER) ---
# robot4 = Drinkbot4()
# robot4.base = ROBOT_BASE_POSES["R4_SERVER"]
# robot4.add_to_env(env)

# Store all robots in a dictionary for easy access
# ROBOTS = {
#     "R1": robot1, 
#     "R2": robot2, 
#     "R3": robot3, 
#     "R4": robot4
# }


# ----------------------------------------------------
# IV. ENVIRONMENT OBJECT DEFINITION
# ----------------------------------------------------

# --- 1. Glass Mesh ---
glass_posex = 0.55
glass_posey = 0.20
# Define the absolute pose of the glass (relative to the world frame)
glass_pose = SE3(glass_posex, glass_posey, GLASS_HEIGHT/2)
wine_glass = Cylinder(radius=GLASS_RADIUS, length=GLASS_HEIGHT,
                     pose=glass_pose, color=[0.85, 0.65, 0.35, 0.8])
env.add(wine_glass)


# --- 2. Ice Dispenser ---
DISP_X = 0.5
DISP_Y = 0.25
DISP_Z = 0.25
DISP_SIZE = [DISP_X, DISP_Y, DISP_Z]
dispenser_pose_x = 0.6
dispenser_pose_y = -0.2
dispenser_pose = SE3(dispenser_pose_x, dispenser_pose_y, DISP_SIZE[2] / 2.0)
ice_dispenser = Cuboid(scale=[DISP_X, DISP_Y, DISP_Z], pose=dispenser_pose, color=[0.2, 0.4, 1.0, 1.0])
env.add(ice_dispenser)


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


# --- 6. Floor and walls ---

wall_height = 2.5      # metres
wall_thickness = 0.05  # metres

# --- Floor ---
floor = Cuboid(scale=[5, 3, 0.02],
               color=[0.25, 0.3, 0.35, 1],
               pose=SE3(0, 0, 0.01))   # raised slightly to avoid flicker
env.add(floor)

# --- Back Wall (5m long) ---
back_wall = Cuboid(scale=[5, wall_thickness, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(0, -1.5, wall_height/2))
env.add(back_wall)

# --- Left Wall (3m long) ---
left_wall = Cuboid(scale=[wall_thickness, 3, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(-2.5, 0, wall_height/2))
env.add(left_wall)

# --- Right Wall (3m long) ---
right_wall = Cuboid(scale=[wall_thickness, 3, wall_height],
                    color=[0.85, 0.85, 0.9, 1],
                    pose=SE3(2.5, 0, wall_height/2))
env.add(right_wall)

# --- 7. Futuristic Tables with 3 Horizontal Wrap LEDs on Table 2 ---

# Common colors
base_color  = [0.1, 0.1, 0.15, 1]     # dark graphite / base
top_color   = [0.0, 0.6, 0.8, 1]      # neon cyan top
top_glow_color = [0.0, 0.8, 1.0, 0.3] # semi-transparent top glow
led_color   = [0.0, 0.8, 1.0, 0.6]    # bright cyan LED
led_height  = 0.05                     # LED thickness
led_offset  = 0.01                     # slightly above floor
led_margin  = 0.02                     # tiny extension beyond table edges

# --- Table 1 (large table, unchanged) ---
table1_length = 3.0
table1_width  = 0.75
table1_height = 1.0
table1_offset_from_wall = 0.5
table1_center_y = -1.5 + wall_thickness + table1_offset_from_wall + table1_width / 2

# Base
table1_base = Cuboid(scale=[table1_length, table1_width, table1_height - 0.05],
                     color=base_color,
                     pose=SE3(0, table1_center_y, (table1_height - 0.05)/2))
env.add(table1_base)

# Tabletop
table1_top = Cuboid(scale=[table1_length, table1_width, 0.05],
                    color=top_color,
                    pose=SE3(0, table1_center_y, table1_height - 0.025))
env.add(table1_top)

# Top glow
table1_glow = Cuboid(scale=[table1_length*1.05, table1_width*1.05, 0.02],
                     color=top_glow_color,
                     pose=SE3(0, table1_center_y, table1_height - 0.015))
env.add(table1_glow)

# Base LED (slightly wider than table)
table1_led = Cuboid(scale=[table1_length + led_margin*2, table1_width + led_margin*2, led_height],
                    color=led_color,
                    pose=SE3(0, table1_center_y, (led_height/2)+led_offset))
env.add(table1_led)

# --- Table 2 (smaller table with horizontal wrap LEDs) ---
table2_length = 1.5
table2_width  = 0.7
table2_height = 1.0
table2_spacing = 1.0
table2_center_y = table1_center_y + (table1_width/2) + table2_spacing + (table2_width/2)

# Base
table2_base = Cuboid(scale=[table2_length, table2_width, table2_height - 0.05],
                     color=base_color,
                     pose=SE3(0, table2_center_y, (table2_height - 0.05)/2))
env.add(table2_base)

# Tabletop
table2_top = Cuboid(scale=[table2_length, table2_width, 0.05],
                    color=top_color,
                    pose=SE3(0, table2_center_y, table2_height - 0.025))
env.add(table2_top)

# Top glow
table2_glow = Cuboid(scale=[table2_length*1.05, table2_width*1.05, 0.02],
                     color=top_glow_color,
                     pose=SE3(0, table2_center_y, table2_height - 0.015))
env.add(table2_glow)

# Base LED (slightly wider than table)
table2_led = Cuboid(scale=[table2_length + led_margin*2, table2_width + led_margin*2, led_height],
                    color=led_color,
                    pose=SE3(0, table2_center_y, (led_height/2)+led_offset))
env.add(table2_led)

# --- Horizontal wrap-around LED rings on Table 2 ---
num_wraps = 3
wrap_height = led_height
# Evenly spaced from top of base LED to just below tabletop
wrap_spacing = (table2_height - 0.05 - led_height) / (num_wraps + 1)

for i in range(1, num_wraps+1):
    wrap_z = led_height + wrap_spacing * i  # start from top of base LED
    wrap_ring = Cuboid(
        scale=[table2_length + led_margin*2, table2_width + led_margin*2, wrap_height],
        color=led_color,
        pose=SE3(0, table2_center_y, wrap_z)
    )
    env.add(wrap_ring)


# ----------------------------------------------------
# V. GRIPPER VISUALS & LOGIC (Needs Refactoring for Multi-Robot)
# ----------------------------------------------------

# --- Gripper Visual Meshes (Attached to Robot 1) ---
left_finger_r1 = Cuboid([FINGER_LENGTH, FINGER_THICKNESS, FINGER_HEIGHT], color=[0.2, 0.8, 0.2, 1])
right_finger_r1 = Cuboid([FINGER_LENGTH, FINGER_THICKNESS, FINGER_HEIGHT], color=[0.2, 0.8, 0.2, 1])
env.add(left_finger_r1)
env.add(right_finger_r1)

# NOTE TO USER: When you add R2, R3, R4, you will need a separate set of finger meshes for each
# e.g., left_finger_r2, right_finger_r2, etc. and associated logic.

def _finger_transform(tcp, side, gap):
    """Calculates the world transform for a finger based on the robot's TCP."""
    y = gap / 2.0 if side == "left" else -gap / 2.0
    # Apply offsets from TCP
    return tcp * SE3(-FINGER_BACK_OFFSET, y, FINGER_Z_OFFSET) * SE3.Ry(np.pi)

def _update_fingers(robot, left_finger_mesh, right_finger_mesh, finger_gap, q=None):
    """
    UPDATED: Now takes robot, finger meshes, and gap as arguments for generalization.
    Updates the position of the visual gripper meshes based on the robot's end-effector.
    """
    # NOTE: Your current logic uses a global variable for _finger_gap, which is problematic for multi-robot.
    # The new version should pass the state/robot object, but for now, we'll keep the original logic 
    # to maintain functionality for R1.

    if robot is not robot1:
        # PLACEHOLDER: Logic for other robots
        return
    
    global _finger_gap_r1
    if q is None:
        q = robot.q
    
    # Calculate the Tool Center Point (TCP) in the world frame
    tcp = robot.fkine(q) 
    
    # Apply the transform to the visual meshes
    left_finger_mesh.T = _finger_transform(tcp, "left", _finger_gap_r1).A
    right_finger_mesh.T = _finger_transform(tcp, "right", _finger_gap_r1).A

def gripper_open(robot, steps=20):
    """Opens the gripper visuals for the specified robot."""
    if robot is not robot1: return # PLACEHOLDER
    
    global _finger_gap_r1
    start = _finger_gap_r1
    
    for g in np.linspace(start, FINGER_GAP_OPEN, steps):
        _finger_gap_r1 = g
        _update_fingers(robot1, left_finger_r1, right_finger_r1, _finger_gap_r1)
        env.step(SIM_STEP_TIME)

def gripper_close(robot, move_mesh=True, steps=20):
    """
    Closes the gripper and optionally attaches the carried object (wine_glass) 
    to the end-effector midpoint (for R1).
    """
    if robot is not robot1: return # PLACEHOLDER
        
    global _finger_gap_r1
    start = _finger_gap_r1
    target = FINGER_GAP_CLOSED
    
    # Get final midpoint for glass attachment
    tcp_now = robot.fkine(robot.q)
    left_final = _finger_transform(tcp_now, "left", target).t
    right_final = _finger_transform(tcp_now, "right", target).t
    midpoint_final = (left_final + right_final) / 2.0
    orig_R = wine_glass.T[0:3, 0:3].copy() # Preserve initial rotation
    
    for i, g in enumerate(np.linspace(start, target, steps)):
        _finger_gap_r1 = g
        _update_fingers(robot1, left_finger_r1, right_finger_r1, _finger_gap_r1)
        
        if move_mesh:
            # Gradually move the held object (wine_glass) towards the final grasp midpoint
            alpha = (i + 1) / steps
            wine_glass.T[0:3, 3] = (1 - alpha) * wine_glass.T[0:3, 3] + alpha * midpoint_final
            wine_glass.T[0:3, 0:3] = orig_R
            
        env.step(SIM_STEP_TIME)

# Initial update of the fingers
_update_fingers(robot1, left_finger_r1, right_finger_r1, _finger_gap_r1)


# ----------------------------------------------------
# VI. MOVEMENT HELPERS (IK & Trajectory)
# ----------------------------------------------------

def wrap_to_near(q_goal, q_ref):
    """Wraps joint angles to minimize angular distance from a reference configuration."""
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi

def solve_ik(robot, T_target, q0):
    """
    UPDATED: Now takes robot as argument.
    Solves Inverse Kinematics for a given robot and target pose.
    """
    # NOTE: If you replace this with the find_ikine method from Drinkbot class, 
    # it handles q0 guess and limits better. Sticking to current logic for now.
    sol = robot.ikine_LM(T_target, q0=q0, mask=[1,1,1,1,1,1], joint_limits=True)
    return sol.q if sol.success else None

# --- Transformation for Carried Object (Glass) ---
GLASS_OFFSET = SE3(0, 0, GLASS_HEIGHT / 2.0) # Offset from EE to the center of the glass base
GLASS_ROT = SE3.Ry(np.pi / 2)                # Required rotation for the glass

def move_joint_traj(robot, q_start, q_goal, steps=TRAJ_STEPS, carry_mesh=None):
    """
    UPDATED: Now takes robot as argument.
    Executes a smooth joint trajectory, optionally carrying a mesh object.
    """
    if q_goal is None:
        print(f"[{robot.name}] IK failed for trajectory, staying at start pose.")
        return q_start 
        
    q_goal = wrap_to_near(q_goal, q_start)
    
    # Generate and step through the joint trajectory
    for q in rtb.jtraj(q_start, q_goal, steps).q:
        robot.q = q
        
        # NOTE: This call must be refactored to handle the correct gripper/gap for each robot
        _update_fingers(robot1, left_finger_r1, right_finger_r1, _finger_gap_r1, q) 
        
        if carry_mesh is not None:
            # Update the mesh pose relative to the robot's TCP
            T_tcp = robot.fkine(q)
            carry_mesh.T = (T_tcp * GLASS_OFFSET * GLASS_ROT).A
            
        env.step(SIM_STEP_TIME)
        
    return q_goal

def cartesian_place_and_release_safe(robot, q_current, tcp_orientation_SE3, place_xy, final_z,
                                     lift_after=0.08, wrist_clearance=0.03, final_offset_z=0.015):
    """
    UPDATED: Now takes robot as argument.
    Executes a single joint-space trajectory descent for smoother place and release.
    (This function handles placing the GLASS carried by R1).
    """
    # Initial TCP configuration
    T_tcp_now = robot.fkine(q_current)
    start_pos = T_tcp_now.t
    
    # Calculate the desired end position of the robot's TCP (above the final glass placement)
    tcp_final_z = final_z + GLASS_OFFSET.t[2]
    start_z = start_pos[2]
    approach_z = max(start_z, tcp_final_z + wrist_clearance)

    # 1. Approach Trajectory
    T_start = SE3(place_xy[0], place_xy[1], approach_z) * tcp_orientation_SE3
    T_end = SE3(place_xy[0], place_xy[1], tcp_final_z + wrist_clearance) * tcp_orientation_SE3
    
    q_start = solve_ik(robot, T_start, q_current)
    q_end = solve_ik(robot, T_end, q_start)

    if q_start is not None and q_end is not None:
        q_now = move_joint_traj(robot, q_start, q_end, steps=50, carry_mesh=wine_glass)
    else:
        q_now = q_current

    # 2. Final Descent before release
    T_final_tcp = SE3(place_xy[0], place_xy[1], tcp_final_z + final_offset_z) * tcp_orientation_SE3
    sol_final = robot.ikine_LM(T_final_tcp, q0=q_now, mask=[1,1,1,1,1,1], joint_limits=True)
    if sol_final and sol_final.success:
        q_now = move_joint_traj(robot, q_now, sol_final.q, steps=10, carry_mesh=wine_glass)

    # 3. Release object
    gripper_open(robot, steps=20) # Open the gripper
    
    # 4. Ensure glass fully settles at final z (detaches from robot)
    T_final = np.eye(4)
    T_final[0:3,0:3] = wine_glass.T[0:3,0:3] # Maintain orientation
    T_final[0:3,3] = np.array([place_xy[0], place_xy[1], final_z]) # Set final absolute position
    wine_glass.T = T_final

    # 5. Lift after release
    try:
        T_lift_tcp = SE3(place_xy[0], place_xy[1], tcp_final_z + lift_after) * tcp_orientation_SE3
        sol_lift = robot.ikine_LM(T_lift_tcp, q0=q_now, mask=[1,1,1,1,1,1], joint_limits=True)
        if sol_lift and sol_lift.success:
            q_now = move_joint_traj(robot, q_now, sol_lift.q, steps=30)
    except Exception:
        print(f"[{robot.name}] Failed to lift after release.")
        pass

    return q_now

# ----------------------------------------------------
# VII. MULTI-ROBOT POSE DEFINITIONS
# ----------------------------------------------------

# --- A. Shared Poses & Robot 1 (Ice/Glass) Poses ---

# Robot 1 Home Position
q_home_r1 = np.array(robot1._qtest).copy()
robot1.q = q_home_r1
_update_fingers(robot1, left_finger_r1, right_finger_r1, _finger_gap_r1) # Set initial visuals

# Poses for picking up the glass
pre_grasp_r1 = SE3(glass_posex - 0.02, glass_posey, GLASS_HEIGHT / 2.0) * SE3.Ry(np.pi/2)
pickup_pose_r1 = SE3(glass_posex, glass_posey, GLASS_HEIGHT / 2.0) * SE3.Ry(np.pi/2)
lift_pose_r1 = SE3(glass_posex, glass_posey, 0.3) * SE3.Ry(np.pi/2)

# Poses for the ice dispenser interaction
disp_front_x = dispenser_pose_x - DISP_X / 1.65
clearance = 0.02
push_x = disp_front_x - clearance
safe_z = DISP_Z + 0.05

over_dispenser_pose_r1 = SE3(push_x, dispenser_pose_y, safe_z) * SE3.Ry(np.pi/2)
push_pose_r1 = SE3(push_x, dispenser_pose_y, DISP_Z / 4) * SE3.Ry(np.pi/2)
touch_pose_r1 = push_pose_r1 * SE3.Tx(-0.03)

# --- B. Robot 2 (Alcohol) Poses (PLACEHOLDERS) ---
# ALCOHOL_FILL_POSE_R2 = SE3(...) # Target TCP pose for pouring alcohol

# --- C. Robot 3 (Mixers) Poses (PLACEHOLDERS) ---
# MIXER_FILL_POSE_R3 = SE3(...) # Target TCP pose for adding mixers

# --- D. Robot 4 (Server) Poses (PLACEHOLDERS) ---
# TRANSFER_POSE_R4 = SE3(...) # Pose for R4 to take glass from R3
# SERVE_POSE_R4 = SE3(...)    # Final serving pose

# --- E. Critical Handover Zones ---
# The glass must be placed at a defined location for the next robot to access it.
HANDOVER_1_COORDS = (glass_posex, glass_posey) # R1 drops glass, R2 picks it up (using glass start pos for now)
# HANDOVER_2_POSE = SE3(x, y, z) # R2 drops glass, R3 picks it up
# HANDOVER_3_POSE = SE3(x, y, z) # R3 drops glass, R4 picks it up


# ----------------------------------------------------
# VIII. COOPERATIVE SEQUENCE LOGIC (Multi-Robot Task Execution)
# ----------------------------------------------------

q_now_r1 = q_home_r1.copy()
# q_now_r2 = q_home_r2.copy() # PLACEHOLDER
# q_now_r3 = q_home_r3.copy() # PLACEHOLDER
# q_now_r4 = q_home_r4.copy() # PLACEHOLDER

print("=============================================")
print(">>> STARTING SEQUENCE 1: R1 (Glass & Ice) <<<")
print("=============================================")


# --- STEP 1: R1 Picks Up Glass ---
# 1) Approach pre-grasp
q_pre_r1 = solve_ik(robot1, pre_grasp_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_pre_r1)
# 2) Pick up glass
q_pick_r1 = solve_ik(robot1, pickup_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_pick_r1)
# 3) Grasp and Lift
gripper_close(robot1, move_mesh=True, steps=30)
print("[R1] Glass secured and lifted.")
q_lift_r1 = solve_ik(robot1, lift_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_lift_r1, carry_mesh=wine_glass)


# --- STEP 2: R1 Dispenses Ice ---
# 4) Move safely over dispenser (approach)
q_over_r1 = solve_ik(robot1, over_dispenser_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_over_r1, carry_mesh=wine_glass)
# 5) Descend straight to push height (simulates pressing the dispenser lever)
q_push_r1 = solve_ik(robot1, push_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_push_r1, carry_mesh=wine_glass)
# 6) Retract slightly (simulates pressing the lever)
q_touch_r1 = solve_ik(robot1, touch_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_touch_r1, carry_mesh=wine_glass)
print("[R1] Ice dispenser activated (simulated).")
# 7) Return to lift pose safely over dispenser
q_over_back_r1 = solve_ik(robot1, over_dispenser_pose_r1, q_now_r1)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_over_back_r1, carry_mesh=wine_glass)
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_lift_r1, carry_mesh=wine_glass)


# --- STEP 3: R1 Places Glass for R2 (Alcohol) ---
# Move to Handover 1 approach pose
q_back_pick_r1 = solve_ik(robot1, pickup_pose_r1, q_now_r1) # Using pickup pose as temporary handover 1 approach
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_back_pick_r1, carry_mesh=wine_glass)

# 8) Smooth place and release (at Handover 1 location)
tcp_orientation = SE3.Ry(np.pi/2)
q_now_r1 = cartesian_place_and_release_safe(
    robot1, q_now_r1, tcp_orientation,
    HANDOVER_1_COORDS,
    final_z=GLASS_HEIGHT / 2.0,
    lift_after=0.08,
    wrist_clearance=0.03,
    final_offset_z=0.015
)
print("[R1] Glass placed at Handover 1 (Ice added).")
# 9) R1 returns home
q_now_r1 = move_joint_traj(robot1, q_now_r1, q_home_r1)

print("=============================================")
print(">>> SEQUENCE 1 COMPLETE. R1 is in home pose. <<<")
print("=============================================")

# Pause to visually confirm the end of the sequence
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
# IX. EXECUTION & HOLD
# ----------------------------------------------------

env.hold()
