import time
import numpy as np
from spatialmath import SE3
from math import pi

# --- JOINT INITIAL GUESSES ---
R1_GUESSES = {
    "HOME": np.deg2rad(np.array([0., 0., 0., 0., 0., 0.])),
    "GLASS_APPROACH": np.deg2rad(np.array([0., 40.68, 16.82, 9.09, 0., 0.])),
    "GLASS_PICKUP": np.deg2rad(np.array([0., 62.27, 0.45, 10.91, 0., 0.])),
    "LIFT_CLEAR": np.deg2rad(np.array([0., 39.55, -11.82, 10.91, 0., 0.])),
    "HANDOFF": np.deg2rad(np.array([-180., 92.95, -61.93, 10.91, 0., 0.])),
}
R2_GUESSES = {
    "HOME": np.deg2rad(np.array([-90., 0., 0., 0., 0., 0.])),
    "PICKUP_DRINK": np.deg2rad(np.array([-74.207, 141.295, -31.751, 9.875, 103.964, -24.255])),
    "PRE_POUR_AWAY": np.deg2rad(np.array([0, 142.39, -64.67, 0, 66.05, -77.03])),
    "POUR_HOVER": np.deg2rad(np.array([-0.55, 133.68, -32.24, -0.57, 104.09, -89.86])),
    "POUR_DRINK": np.deg2rad(np.array([0, 116, -31, 0, 240, -90])),
    "PLACE_GLASS": np.deg2rad(np.array([0, 25.495, 174.686, 0, -208, 0]))
}
R3_GUESSES = {
    "HOME": np.deg2rad(np.array([180., 0., 0., 0., 0., 0.])),
    "PICKUP_YELLOW": np.deg2rad(np.array([0, 47, 73, -32, 91, 0])),
    "PICKUP_GREEN": np.deg2rad(np.array([-20, 47, 65, -30, 89, 0])),
    "PICKUP_BLUE": np.deg2rad(np.array([-34, 53, 65, -30, 89, 0])),
    "DEPOSIT_INGREDIENTS": np.deg2rad(np.array([0, 0, 0, 0, 0, 0]))
}

def run_robot1_sequence1(controller, robot1, scene):
    """Executes the pick-and-place sequence for Robot 1 (Glassbot)."""
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
    r1_target = scene.glass_poses[glass_index] @ SE3.Ry(pi)
    r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["GLASS_APPROACH"], 
                                             ignore_var="z", ignore_rotation=False, hover_max=0.5)
    if not success:
        print("❌ [R1] Failed to find hover path to glass.")
        return

    controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)

    # Step 2: Move down to the grasping position
    print("\n[R1] Moving to grasp position...")
    target_grasp_pose = scene.glass_poses[glass_index] @ SE3.Tz(scene.glass_height / 2) @ SE3.Ry(pi)
    controller.move_cartesian(robot1, robot1.q, target_grasp_pose, 50)
    controller.print_pose(robot1, "R1 at Grasp Position")
    
    # Step 3: Pick up the object
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
    if not success:
        print("❌ [R1] Failed to find path to workstation.")
        return

    controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)
    controller.print_pose(robot1, "R1 at hover before placing glass")
    controller.move_cartesian(robot1, robot1.q, r1_target, 50)
    controller.drop_object(robot1)

    # Step 6: Move glassbot back to home position
    print("\n[R1] raising EE up...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.3)
    controller.move_cartesian(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted EE")
    controller.animate_trajectory(robot1, robot1.q, np.zeros(6), steps=60)


def run_robot2_sequence1(controller, robot2, scene):
    """Executes the drink pouring sequence for Robot 2 (DrinkBot)."""
    print("\n" + "="*70)
    print(">>> ROBOT 2: FETCHING AND POURING DRINK <<<")
    print("="*70 + "\n")

    drink_index = 3 
    target_drink = scene.drink_objects[drink_index]

    # Step 7: Approach and move to the drink
    print("\n[R2] Approaching drink...")
    target_r2_pose = scene.drink_poses[drink_index] @ SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2)
    hover_q_r2, success = controller.find_ikine(robot2, target_r2_pose, R2_GUESSES["PICKUP_DRINK"], "y", False, 0.5)
    if not success:
        print("❌ [R2] Failed to find hover path to drink.")
        return

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

    if not success:
        print("❌ [R2] Unable to find a valid path to the pouring position.")
        return

    controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    controller.print_pose(robot2, "R2 Ready to Pour")
    
    # Step 12: Pour the drink by rotating the wrist
    print("\n[R2] Pouring...")
    pour_q = robot2.q.copy()
    pour_q[4] += np.deg2rad(115)
    controller.animate_trajectory(robot2, robot2.q, pour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished Pouring")

    # Step 13: Rotate the drink back up
    print("\n[R2] Un-pouring...")
    unpour_q = robot2.q.copy()
    unpour_q[4] -= np.deg2rad(115)
    controller.animate_trajectory(robot2, robot2.q, unpour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished un-pouring")

    # Step 14: Return the drink
    print("\n[R2] Returning drink to shelf...")
    controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
    controller.move_cartesian(robot2, robot2.q, target_r2_pose, 50)
    controller.drop_object(robot2)

    # Return home
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["HOME"], steps=60)

def run_robot3_sequence1(controller, robot3, scene):
    """Executes the ingredient adding sequence for Robot 3 (IngredientBot)."""
    print("\n" + "="*70)
    print(">>> ROBOT 3: ADDING INGREDIENT <<<")
    print("="*70 + "\n")

    # Configuration
    CUBE_INDEX = 8
    HOVER_HEIGHT_R3 = 0.08
    DROP_HEIGHT_R3 = 0.2
    PLACEMENT_DEPTH = 0.05
    X_GLASS, Y_GLASS, Z_GLASS = -0.9, 0, 1.1

    # Target Objects & Poses
    cube_target = scene.cube_objects[CUBE_INDEX]
    VERTICAL_ORIENTATION = SE3.Rx(pi)
    PICK_POSE = scene.cube_poses[CUBE_INDEX] @ VERTICAL_ORIENTATION 
    HOVER_POSE = PICK_POSE @ SE3.Tz(-HOVER_HEIGHT_R3)
    PLACE_POSE = SE3(X_GLASS, Y_GLASS, Z_GLASS) @ SE3.Tz(PLACEMENT_DEPTH) @ VERTICAL_ORIENTATION
    DROP_POSE = PLACE_POSE @ SE3.Tz(DROP_HEIGHT_R3)

    # Step 1: Move to initial hover position
    print("\n[R3] Approaching cube...")
    controller.animate_trajectory(robot3, robot3.q, R3_GUESSES["PICKUP_YELLOW"], steps=60)
    controller.print_pose(robot3, f"R3 at Hover before Cube {CUBE_INDEX}")

    # Step 2: Move down to the Pick Pose
    print("\n[R3] Moving to pick position...")
    controller.move_cartesian(robot3, robot3.q, PICK_POSE, 50)
    controller.print_pose(robot3, "R3 at Pick Position")

    # Step 3: Pick up the object
    controller.pickup_object(robot3, cube_target)

    # Step 4: Retract back to Hover Pose
    print("\n[R3] Retracting to hover...")
    controller.move_cartesian(robot3, robot3.q, HOVER_POSE, 50)
    controller.print_pose(robot3, "R3 Retracted")

    # Step 5: Swing around to the drop hover position
    print("\n[R3] Swinging to drop area...")
    controller.animate_trajectory(robot3, robot3.q, R3_GUESSES["DEPOSIT_INGREDIENTS"], steps=80)
    controller.print_pose(robot3, "R3 at Drop Hover")

    # Step 6: Move down to the Place Pose inside glass
    print("\n[R3] Placing cube in glass...")
    controller.move_cartesian(robot3, robot3.q, PLACE_POSE, 50)
    controller.print_pose(robot3, "R3 at Place Position")

    # Step 7: Release the object (Corrected from release_object)
    controller.drop_object(robot3) 
    print(f"Cube {CUBE_INDEX} released.")

    # Step 8: Retract back to Drop Hover Pose
    print("\n[R3] Retracting from glass...")
    controller.move_cartesian(robot3, robot3.q, DROP_POSE, 50)

    # Step 9: Move to home position
    print("\n[R3] Returning to home...")
    controller.animate_trajectory(robot3, robot3.q, R3_GUESSES["HOME"], steps=60)
    controller.print_pose(robot3, "R3 at Home")