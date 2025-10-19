import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid, Box
from robot_helpers import RobotController
from math import pi

from Drinkbot import Drinkbot
from IngredientBot import IngredientBot
from Glassbot import Glassbot
from Serverbot import Serverbot
from EnvironmentSetup import Scene

env = swift.Swift()
env.launch(realtime=True)

env.set_camera_pose([0, 3, 4], [0, 0, 0.5])

scene = Scene(env)
controller = RobotController(env, scene)

# ----------------------------------------------------
# ROBOT CREATION & PLACEMENT
# ----------------------------------------------------

# Robot 1: Glass & Ice Handler
robot1 = Glassbot()
robot1.base = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

# Robot 2: Alcohol Pourer
robot2 = IngredientBot()
robot2.q = robot2.home_q
robot2.base = scene.ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

# Robot 3: Mixer Adder
robot3 = Drinkbot()
robot3.q = robot3.home_q
robot3.base = scene.ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

# Robot 4: Server (placeholder)
robot4 = Serverbot()
robot4.base = scene.ROBOT_BASE_POSES["R4_SERVER"]
robot4.q = np.array([0,-pi/2,0,0,0,0])
robot4.add_to_env(env)

# ============================================================================
# UTILITY FUNCTIONS (Removed find_closest_cube_index as requested)
# ============================================================================

# ============================================================================
# JOINT INITIAL GUESSES
# ============================================================================

# R1 (Glassbot)
R1_GUESSES = {
    "HOME": np.deg2rad(np.array([0., 0., 0., 0., 0., 0.])),
    "GLASS_APPROACH": np.deg2rad(np.array([0., 40.68, 16.82, 9.09, 0., 0.])),
    "GLASS_PICKUP": np.deg2rad(np.array([0., 62.27, 0.45, 10.91, 0., 0.])),
    "LIFT_CLEAR": np.deg2rad(np.array([0., 39.55, -11.82, 10.91, 0., 0.])),
    "ICE_MACHINE": np.deg2rad(np.array([-87.95, 52.05, 3.52, 10.91, 0., 0.])),
    "HANDOFF": np.deg2rad(np.array([-180., 92.95, -61.93, 10.91, 0., 0.])),
}

# R2 (Drinkbot)
R2_GUESSES = {
    "HOME": robot2.q.copy(),
    "PICKUP_DRINK": np.deg2rad(np.array([-74.207, 141.295, -31.751, 9.875, 103.964, -24.255])), # Position to pick up drink
    "PLACE_GLASS": np.deg2rad(np.array([0, 25.495, 174.686, 0, -208, 0])) # Position to place drink
}

# R3 (Drinkbot)
R3_GUESSES = {
    "HOME": robot3.q.copy(),
    "PICKUP_YELLOW": np.deg2rad(np.array([0, 47, 73, -32, 91, 0])), # Position to pick up Yellow ingredient (Key standardized to ALL CAPS)
    "PICKUP_GREEN": np.deg2rad(np.array([-20, 47, 65, -30, 89, 0])), # Position to pick up Green ingredient
    "PICKUP_BLUE": np.deg2rad(np.array([-34, 53, 65, -30, 89, 0])), # Position to pick up Blue ingredient
    "DEPOSIT_INGREDIENTS": np.deg2rad(np.array([0, 0, 0, 0, 0, 0])), # glass drop off position where i will add ingredients
    "DROP_HOVER": np.deg2rad(np.array([-180, 45, 60, -30, 90, 0])), # New hardcoded guess for drop hover
}

# ============================================================================
# ROBOT 1 SEQUENCE - COMMENTED OUT
# ============================================================================
# ... R1 code remains commented out ...

# ============================================================================
# ROBOT 2 SEQUENCE - COMMENTED OUT
# ============================================================================
# ... R2 code remains commented out ...


# ============================================================================
# ROBOT 3 SEQUENCE - ADD INGREDIENTS TO GLASS
# ============================================================================

# yellow cubes are 0-8 (Index 0 is the closest cube)
# green cubes are 9-17
# blue cubes are 18-26

# Configuration
CUBE_INDEX = 8 # Confirmed reachable cube
HOVER_HEIGHT_R3 = 0.08 # Increased hover height for better clearance
DROP_HEIGHT_R3 = 0.2
PLACEMENT_DEPTH = 0.05 # Decreased placement depth (placing it deeper into the glass)
# CUBE_GRIP_OFFSET removed to prevent clipping the table!

X_GLASS = -1.0
Y_GLASS = 0
Z_GLASS = 1.1

# Target Objects
cube_target = scene.cube_objects[CUBE_INDEX]
GLASS_PLACEMENT_POSE = SE3(X_GLASS, Y_GLASS, Z_GLASS)
VERTICAL_ORIENTATION = SE3.Rx(pi)

# FIND POSE

cube_target_pose = scene.cube_poses[CUBE_INDEX] 
# PICK_POSE now targets the exact top surface of the cube to prevent clipping
PICK_POSE = cube_target_pose @ VERTICAL_ORIENTATION 
HOVER_POSE = PICK_POSE @ SE3.Tz(-HOVER_HEIGHT_R3) # Hover above the pickup pose, using negative Z in the tool frame

glass_target_pose = GLASS_PLACEMENT_POSE
PLACE_POSE = glass_target_pose @ SE3.Tz(PLACEMENT_DEPTH) @ VERTICAL_ORIENTATION
DROP_POSE = PLACE_POSE @ SE3.Tz(DROP_HEIGHT_R3)

# Execute Sequence 
print("\n" + "="*70)
print(f">>> ROBOT R3: PICKING UP CUBE {CUBE_INDEX} <<<")
print("="*70 + "\n")

# Use the known good joint state for the approach hover
hover_q_r3 = R3_GUESSES["PICKUP_YELLOW"] 

# Step 1: Move to initial hover position (Joint space move - R2 equivalent Step 7 approach)
print("\n[R3] Approaching cube hover position...")
controller.animate_trajectory(robot3, robot3.q, hover_q_r3, steps=60)
controller.print_pose(robot3, f"R3 at Hover before Cube {CUBE_INDEX} Pickup")

# Step 2: Move down to the Pick Pose (Cartesian move - R2 equivalent Step 7 down)
print("\n[R3] Moving down to Pick Pose (Cartesian move)...")
controller.move_cartesian(robot3, robot3.q, PICK_POSE, 50)
controller.print_pose(robot3, "R3 at Pick Position")

# Step 3: Pick up the object (R2 equivalent Step 8)
controller.pickup_object(robot3, cube_target)
print(f"Cube {CUBE_INDEX} picked up.")

# Step 4: Retract back to Hover Pose (Cartesian move - R2 equivalent Step 9 retract)
# This uses HOVER_POSE which is guaranteed to be safe and above the table.
print("\n[R3] Retracting to Hover Pose...")
controller.move_cartesian(robot3, robot3.q, HOVER_POSE, 50)
controller.print_pose(robot3, "R3 Retracted")

# Step 5: Swing around to the drop hover position (Hardcoded Joint Space Move - R2 equivalent Step 10)
# This now uses a better guess to turn the base 90 degrees toward the drop zone.
print("\n[R3] Swinging around to drop area...")
drop_q_r3 = R3_GUESSES["DROP_HOVER"]
controller.animate_trajectory(robot3, robot3.q, drop_q_r3, steps=80) # Use more steps for a smoother 'turn around'
controller.print_pose(robot3, "R3 at Drop Hover Position")

# Step 6: Move down to the Place Pose inside glass (Cartesian move - R2 equivalent Step 11 final pour)
print("\n[R3] Moving down to Place Pose inside glass...")
controller.move_cartesian(robot3, robot3.q, PLACE_POSE, 50)
controller.print_pose(robot3, "R3 at Place Position")

# Step 7: Release the object (R2 equivalent Step 12/13 pour/unpour action)
controller.release_object(robot3) 
print(f"Cube {CUBE_INDEX} released into the static drink location.")

# Step 8: Retract back to Drop Hover Pose (Cartesian move - R2 equivalent Step 14 retract)
print("\n[R3] Retracting back to Drop Hover...")
controller.move_cartesian(robot3, robot3.q, DROP_POSE, 50)

# Step 9: Move to a safe 'home' or park position (R2 equivalent final home)
print("\n[R3] Moving to Park Position...")
park_q = R3_GUESSES["HOME"] 
controller.animate_trajectory(robot3, robot3.q, park_q, steps=60)
controller.print_pose(robot3, "R3 at Park Position")

env.hold()

