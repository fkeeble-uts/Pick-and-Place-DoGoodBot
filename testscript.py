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
CUBE_INDEX = 0 # Explicitly set to the closest cube (Yellow, Index 0)
HOVER_HEIGHT_R3 = 0.05
DROP_HEIGHT_R3 = 0.2
PLACEMENT_DEPTH = 0.1

X_GLASS = -0.9
Y_GLASS = 0
Z_GLASS = 1.1

# Target Objects
cube_target = scene.cube_objects[CUBE_INDEX]
GLASS_PLACEMENT_POSE = SE3(X_GLASS, Y_GLASS, Z_GLASS)
VERTICAL_ORIENTATION = SE3.Rx(pi)

# FIND POSE

cube_target_pose = scene.cube_poses[CUBE_INDEX] 
PICK_POSE = cube_target_pose @ VERTICAL_ORIENTATION
HOVER_POSE = PICK_POSE @ SE3.Tz(HOVER_HEIGHT_R3)

glass_target_pose = GLASS_PLACEMENT_POSE
PLACE_POSE = glass_target_pose @ SE3.Tz(PLACEMENT_DEPTH) @ VERTICAL_ORIENTATION
DROP_POSE = PLACE_POSE @ SE3.Tz(DROP_HEIGHT_R3)

# Execute Sequence 
print("\n" + "="*70)
print(f">>> ROBOT R3: PICKING UP CUBE {CUBE_INDEX} <<<")
print("="*70 + "\n")

# --- HARDCODE FIX FOR URGENT PROGRESS CHECK ---
# We are skipping the IK search and using a known, defined joint state 
# to ensure the move works immediately for the initial hover.
hover_q_r3 = R3_GUESSES["PICKUP_YELLOW"] # Use the known good guess for the approach
success = True 

if success:
    # Step 1: Move to initial hover position (Joint space move)
    controller.animate_trajectory(robot3, robot3.q, hover_q_r3, steps=60)
    controller.print_pose(robot3, f"R3 at Hover before Cube {CUBE_INDEX} Pickup (Hardcoded)")

    # Step 2: Move down to the Pick Pose (Reverting to Cartesian move for vertical drop)
    print("\n[R3] Moving down to Pick Pose (using Cartesian move)...")
    
    # Use the robust controller.move_cartesian for the short, vertical, straight-line drop
    # This replaces the failing manual IK/joint move combination.
    controller.move_cartesian(robot3, robot3.q, PICK_POSE, 50)
    controller.print_pose(robot3, "R3 at Pick Position")

    # Step 3: Pick up the object
    controller.pickup_object(robot3, cube_target)
    print(f"Cube {CUBE_INDEX} picked up.")

    # Step 4: Retract back to Hover Pose (Cartesian motion is safer here since it's only moving up)
    print("\n[R3] Retracting to Hover Pose...")
    controller.move_cartesian(robot3, robot3.q, HOVER_POSE, 50)

    # Step 5: Move to the Drop Pose above the glass 

    # CRITICAL FIX (Temporary Hardcode): Bypassing IK entirely for the transition move.
    drop_q_r3 = R3_GUESSES["DROP_HOVER"]
    success_drop = True

    if success_drop:
        controller.animate_trajectory(robot3, robot3.q, drop_q_r3, steps=80) # Use more steps for a smoother 'turn around'
        controller.print_pose(robot3, "R3 at Drop Pose before Placement (Hardcoded)")

        # Step 6: Move down to the Place Pose (Cartesian motion)
        print("\n[R3] Moving down to Place Pose inside glass...")
        controller.move_cartesian(robot3, robot3.q, PLACE_POSE, 50)

        # Step 7: Release the object
        controller.release_object(robot3, cube_target) 
        print(f"Cube {CUBE_INDEX} released into the static drink location.")

        # Step 8: Retract back to Drop Pose
        print("\n[R3] Retracting to Drop Pose...")
        controller.move_cartesian(robot3, robot3.q, DROP_POSE, 50)

        # Step 9: Move to a safe 'home' or park position
        print("\n[R3] Moving to Park Position...")
        park_q = R3_GUESSES["HOME"] # Using the user-defined park guess
        controller.animate_trajectory(robot3, robot3.q, park_q, steps=60)
        controller.print_pose(robot3, "R3 at Park Position")
        
    else:
        print("FATAL ERROR: Movement failed after pickup due to hardcoded IK bypass not working (unlikely).")

else:
    print("FATAL ERROR: Unable to find IK solution for cube hover pose.")

env.hold()
