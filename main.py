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
robot4.base = scene.ROBOT_BASE_POSES["R4_SERVER"] * SE3.Rx(pi/2) * SE3.Ry(pi/2)
robot4.add_to_env(env)

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
    "PICKUP_DRINK": np.deg2rad(np.array([-74.207, 141.295, -31.751, 9.875, 103.964, -24.255])),  # Position to pick up drink
    "PLACE_GLASS": np.deg2rad(np.array([0, 25.495, 174.686, 0, -208, 0]))    # Position to place drink
}

# R3 (Drinkbot) 
R3_GUESSES = {
    "HOME": robot3.q.copy(),
    "PICKUP_Yellow": np.deg2rad(np.array([0, 47, 73, -32, 91, 0])),  # Position to pick up Yellow ingredient
    "PICKUP_GREEN": np.deg2rad(np.array([-20, 47, 65, -30, 89, 0])),   # Position to pick up Green ingredient
    "PICKUP_BLUE": np.deg2rad(np.array([-34, 53, 65, -30, 89, 0])),  # Position to pick up Blue ingredient
    "DEPOSIT_INGREDIENTS": np.deg2rad(np.array([0, 0, 0, 0, 0, 0]))    # glass drop off position where i will add ingredients
}

# ============================================================================
# ROBOT 1 SEQUENCE - PICK AND PLACE GLASS
# ============================================================================

print("\n" + "="*70)
print(">>> ROBOT 1: PICKING UP GLASS <<<")
print("="*70 + "\n")

glass_index = 3 
target_glass = scene.glass_objects[glass_index]

robot1.q = R1_GUESSES["HOME"]
controller.print_pose(robot1, "R1 at HOME")
time.sleep(0.5)

# Step 1: Move to a hover position above the glass
print("\n[R1] Moving to hover above glass...")
# Target the center of the glass, but with a vertical offset for the TCP
target_hover_pose = scene.glass_poses[glass_index] @ SE3.Tz(0.2) @ SE3.Ry(pi)
q_hover, success = controller.find_ikine(robot1, target_hover_pose, R1_GUESSES["GLASS_APPROACH"])

if success:
    controller.move_to_q(robot1, q_hover, name="Glass Hover")

    # Step 2: Move down to the grasping position
    print("\n[R1] Moving to grasp position...")
    target_grasp_pose = scene.glass_poses[glass_index] @ SE3.Tz(scene.glass_height / 2) @ SE3.Ry(pi)
    controller.move_cartesian(robot1, robot1.q, target_grasp_pose, 50)
    controller.print_pose(robot1, "R1 at Grasp Position")
    
    # Step 3: PICK UP the object
    # This calculates and stores the grasp transform
    controller.pickup_object(robot1, target_glass)
    time.sleep(1) # Simulate gripper closing
    
    # Step 4: Lift the glass vertically
    print("\n[R1] Lifting glass...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.2)
    # The glass will now move correctly because the controller is handling it
    controller.move_cartesian(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted Glass")

'''
# Step 2: Move to glass level
print("\n[R1] Moving to glass pickup position...")
q_now_r1 = controller.move_to_q(robot1, R1_GUESSES["GLASS_PICKUP"], steps=40, name="Glass Pickup", 
                      carry_object=target_glass)
controller.print_pose(robot1, "R1 at GLASS_PICKUP")
time.sleep(0.5)

# Step 3: Simulate gripper closing
print("\n[R1] Closing gripper...")
held_by_r1 = True
time.sleep(0.5)

# Step 4: Lift glass
print("\n[R1] Lifting glass...")
q_now_r1 = controller.move_to_q(robot1, R1_GUESSES["LIFT_CLEAR"], steps=40, name="Lift Clear",
                      carry_object=target_glass)
controller.print_pose(robot1, "R1 at LIFT_CLEAR")
time.sleep(0.5)

# Step 5: Move to ice machine
print("\n[R1] Moving to ice machine...")
q_now_r1 = controller.move_to_q(robot1, R1_GUESSES["ICE_MACHINE"], steps=60, name="Ice Machine",
                      carry_object=target_glass)
controller.print_pose(robot1, "R1 at ICE_MACHINE")
print("[R1] Simulating ice fill...")
time.sleep(1.0)

# Step 6: Move to handoff location
print("\n[R1] Moving to handoff location...")
q_now_r1 = controller.move_to_q(robot1, R1_GUESSES["HANDOFF"], steps=60, name="Handoff",
                      carry_object=target_glass)
controller.print_pose(robot1, "R1 at HANDOFF")
time.sleep(0.5)
'''


# ============================================================================
# ROBOT 2 SEQUENCE - ADD ALCOHOL TO GLASS
# ============================================================================

print("\n" + "="*70)
print(">>> ROBOT 2: MOVING TO DRINK 4 <<<")
print("="*70 + "\n")

print("Glass poses:")
for i in range(len(scene.glass_poses)):
    print(f"Pose of glass {i}:")
    print(scene.glass_poses[i])

# Step 7: Move to drink 4
print("\n[R2] Moving to drink 4...")
target_r2_pose = scene.drink_poses[3] @ SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2)
hover_q_r2, success = controller.find_ikine(robot2, target_r2_pose, R2_GUESSES["PICKUP_DRINK"], "y", False, 0.5)
controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
controller.print_pose(robot2, "R2 at Hover before Drink 4")
if success:
    controller.move_cartesian(robot2, robot2.q, target_r2_pose, 50)
else:
    print("Unable to move robot2 to hover pose")

# 

env.hold()