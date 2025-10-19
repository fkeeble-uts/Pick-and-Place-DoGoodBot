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
    "PRE_POUR_AWAY": np.deg2rad(np.array([0, 142.39, -64.67, 0, 66.05, -77.03])),
    "POUR_HOVER": np.deg2rad(np.array([-0.55, 133.68, -32.24, -0.57, 104.09, -89.86])),
    "POUR_DRINK": np.deg2rad(np.array([0, 116, -31, 0, 240, -90])),
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

'''
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
# Target the center of the glass, but mask off z
r1_target = scene.glass_poses[glass_index] @ SE3.Ry(pi)
r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["GLASS_APPROACH"], 
                                         ignore_var="z", ignore_rotation=False, hover_max=0.5)

if success:
    controller.move_to_q(robot1, r1_q_hover, name="Glass Hover")

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
    controller.move_cartesian(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted Glass")

    # Step 5: Hover the glass over the workstation
    print("\n[R1] bringing glass to workstation...")
    r1_target = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ SE3.Tx(-0.5) @ SE3.Tz(scene.glass_height) @ SE3.Ry(pi)
    r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["HANDOFF"], 
                                             ignore_var="z", ignore_rotation=False, hover_max=0.5)
    
    controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)
    controller.print_pose(robot1, "R1 at hover before placing glass")
    if success:
        controller.move_cartesian(robot1, robot1.q, r1_target, 50)
    else:
        print("Unable to move robot1 to hover pose")
    controller.drop_object(robot1)

    # Step 6: Move glassbot back to home position
    print("\n[R1] raising EE up...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.3)
    controller.move_cartesian(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted Glass")
    controller.animate_trajectory(robot1, robot1.q, np.zeros(6), steps=60)
'''

# ============================================================================
# ROBOT 2 SEQUENCE - ADD ALCOHOL TO GLASS
# ============================================================================
print("\n" + "="*70)
print(">>> ROBOT 2: MOVING TO DRINK 4 <<<")
print("="*70 + "\n")

drink_index = 3 
target_drink = scene.drink_objects[drink_index]

# Step 7: Approach and move to the drink
print("\n[R2] Approaching drink...")
target_r2_pose = scene.drink_poses[drink_index] @ SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2)
hover_q_r2, success = controller.find_ikine(robot2, target_r2_pose, R2_GUESSES["PICKUP_DRINK"], "y", False, 0.5)
controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
controller.move_cartesian(robot2, robot2.q, target_r2_pose, 50)

# Step 8: Pick up the drink
controller.pickup_object(robot2, target_drink)

# Step 9: Retract from the wall
print("\n[R2] Retracting from shelf...")
retract_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.2)
controller.move_cartesian(robot2, robot2.q, retract_pose, 50)
controller.print_pose(robot2, "R2 Retracted")

# Step 10: Move to an intermediate position
print("\n[R2] Swinging around to pouring area...")
controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["PRE_POUR_AWAY"], steps=60)
controller.print_pose(robot2, "R2 at Pre-Pour Position")

# Step 11: Move to the final pouring position
print("\n[R2] Moving to final pour position...")
pour_height = 0.5
pour_pose = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ SE3(-0.6, 0, scene.glass_height + pour_height) @ SE3.Rx(pi/2) @ SE3.Ry(pi/2)
final_q, success = controller.find_ikine(robot2, pour_pose, initial_q_guess=robot2.q)

if success:
    controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    controller.print_pose(robot2, "R2 Ready to Pour")
    
    # Step 12: Pour the drink by rotating the wrist (joint 5)
    print("\n[R2] Pouring...")
    pour_q = robot2.q.copy()
    pour_q[4] += np.deg2rad(115) # Rotate joint 5 by 90 degrees to pour
    controller.animate_trajectory(robot2, robot2.q, pour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished Pouring")

    # Step 13: Rotate the drink back up by rotating the wrist (joint 5)
    print("\n[R2] Rotating...")
    unpour_q = robot2.q.copy()
    unpour_q[4] -= np.deg2rad(115) # Rotate joint 5 by 90 degrees to unpour
    controller.animate_trajectory(robot2, robot2.q, unpour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished rotating drink")

    # Step 14: Return the drink
    print("\n[R2] Approaching drink return...")
    target_r2_pose = scene.drink_poses[drink_index] @ SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2)
    hover_q_r2, success = controller.find_ikine(robot2, target_r2_pose, R2_GUESSES["PICKUP_DRINK"], "y", False, 0.5)
    controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
    controller.move_cartesian(robot2, robot2.q, target_r2_pose, 50)

    controller.drop_object(robot2)

    # Return home
    home_q = R2_GUESSES["HOME"]
    controller.animate_trajectory(robot2, robot2.q, home_q, steps=60)

else:
    print("Unable to find a valid path to the pouring position.")

env.hold()