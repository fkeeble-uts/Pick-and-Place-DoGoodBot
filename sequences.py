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
    "PICKUP_DRINK": np.deg2rad(np.array([-64.77, 141.16, -61.41, 27.03, 69.68, -79.95])),
    "PRE_POUR_AWAY": np.deg2rad(np.array([0, 142.39, -64.67, 0, 66.05, -77.03])),
    "POUR_HOVER": np.deg2rad(np.array([-0.55, 133.68, -32.24, -0.57, 104.09, -89.86])),
    "GLASS_HOVER": np.deg2rad(np.array([-180, 15, 112, 0, 94, 0])),
    "GLASS_DROP_HOVER": np.deg2rad(np.array([0.49, 19.42, 128.59, 0.0, 109.17, -179.51])),
    "GLASS_HANDOVER_PICKUP": np.deg2rad(np.array([0, 138.77, -16.77, -0.0, -155.53, -90.43])),
    "GLASS_HANDOVER_HOVER" : np.deg2rad(np.array([89.57, 128.63, -16.52, 0.0, -145.15, -90.43]))
}
R3_GUESSES = {
    "HOME": np.deg2rad(np.array([180., 0., 0., 0., 0., 0.])),
    "PICKUP_YELLOW": np.deg2rad(np.array([0, 47, 73, -32, 91, 0])),
    "PICKUP_GREEN": np.deg2rad(np.array([-20, 47, 65, -30, 89, 0])),
    "PICKUP_BLUE": np.deg2rad(np.array([-34, 53, 65, -30, 89, 0])),
    "PICKUP_HOVER": np.deg2rad(np.array([2.45, 35.93, 56.66, -2.59, 90, -87.55])),
    "DEPOSIT_INGREDIENTS": np.deg2rad(np.array([168.07, 32.05, 68.88, -10.93, 90.0, -101.93])),
    "DROP_INGREDIENTS" : np.deg2rad(np.array([25.4, -35.93, -56.66, 2.59, -90, -64.6]))
}

def run_sequence1(controller, robot1, scene):
    """Executes the pick-and-place sequence for Robot 1 (Glassbot)."""
    print("\n" + "="*70)
    print(">>> ROBOT 1: PICKING UP GLASS <<<")
    print("="*70 + "\n")

    glass_index = 3 
    target_glass = scene.glass_objects[glass_index]

    robot1.q = R1_GUESSES["HOME"]
    controller.print_pose(robot1, "R1 at HOME")

    # Step 1: Move to a hover position above the glass
    print("\n[R1] Moving to hover above glass...")
    r1_target = scene.glass_poses[glass_index] @ SE3.Tz(scene.glass_height/2) @ SE3.Ry(pi)
    r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["GLASS_APPROACH"], 
                                             ignore_var="z", ignore_rotation=False, hover_max=0.5)
    if not success:
        print("❌ [R1] Failed to find hover path to glass.")
        return

    controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)

    # Step 2: Move down to the grasping position
    print("\n[R1] Moving to grasp position...")
    target_grasp_pose = scene.glass_poses[glass_index] @ SE3.Tz(scene.glass_height / 2) @ SE3.Ry(pi)
    controller.move_rmrc(robot1, robot1.q, target_grasp_pose, 50)
    controller.print_pose(robot1, "R1 at Grasp Position")
    
    # Step 3: Pick up the object
    controller.pickup_object(robot1, target_glass)
    
    # Step 4: Lift the glass vertically
    print("\n[R1] Lifting glass...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.15)
    controller.move_rmrc(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted Glass")

    # Step 5: Hover the glass over the workstation
    print("\n[R1] bringing glass to workstation...")
    r1_target = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ SE3(-0.5, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ SE3.Ry(pi)
    r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["HANDOFF"], 
                                             ignore_var="z", ignore_rotation=False, hover_max=0.5)
    if not success:
        print("❌ [R1] Failed to find path to workstation.")
        return
    controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)
    controller.print_pose(robot1, "R1 at hover before placing glass")

    # Step 6: Drop the glass
    controller.move_rmrc(robot1, robot1.q, r1_target, 50)
    controller.drop_object(robot1)

    # Step 7: Move glassbot back to home position
    print("\n[R1] raising EE up...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.25)
    controller.move_rmrc(robot1, robot1.q, lift_pose, 50)
    controller.print_pose(robot1, "R1 Lifted EE")
    controller.animate_trajectory(robot1, robot1.q, np.zeros(6), steps=60)


def run_sequence2(controller, robot2, scene):
    """Executes the drink pouring sequence for Robot 2 (DrinkBot)."""
    print("\n" + "="*70)
    print(">>> ROBOT 2: FETCHING AND POURING DRINK <<<")
    print("="*70 + "\n")

    drink_index = 3 
    target_drink = scene.drink_objects[drink_index]

    # Step 1: Approach and move to the drink
    print("\n[R2] Approaching drink...")
    target_r2_pose = scene.drink_poses[drink_index] @ SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2)
    hover_q_r2, success = controller.find_ikine(robot2, target_r2_pose, R2_GUESSES["PICKUP_DRINK"], "y", False, 0.5)
    if not success:
        print("❌ [R2] Failed to find hover path to drink.")
        return
    controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
    controller.print_pose(robot2, "R2 Hovering in front of glass")
    controller.move_rmrc(robot2, robot2.q, target_r2_pose, 50)
    controller.print_pose(robot2, "R2 Gripping glass")

    # Step 2: Pick up the drink
    controller.pickup_object(robot2, target_drink)

    # Step 3: Retract from the wall
    print("\n[R2] Retracting from shelf...")
    retract_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.2)
    controller.move_rmrc(robot2, robot2.q, retract_pose, 50)
    controller.print_pose(robot2, "R2 Retracted")

    # Step 4: Move to an intermediate position
    print("\n[R2] Swinging around to pouring area...")
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["PRE_POUR_AWAY"], steps=60)
    controller.print_pose(robot2, "R2 at Pre-Pour Position")

    # Step 5: Move to the final pouring position
    print("\n[R2] Moving to final pour position...")
    pour_height = 0.35
    pour_pose = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ SE3(-0.6, 0, scene.glass_height + pour_height) @ SE3.Rx(pi/2) @ SE3.Ry(pi/2)
    final_q, success = controller.find_ikine(robot2, pour_pose, initial_q_guess=robot2.q)

    if not success:
        print("❌ [R2] Unable to find a valid path to the pouring position.")
        return

    controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    controller.print_pose(robot2, "R2 Ready to Pour")
    
    # Step 6: Pour the drink by rotating the wrist
    print("\n[R2] Pouring...")
    print(f"Robot q before pouring: {np.rad2deg(robot2.q)}")
    pour_q = robot2.q.copy()
    pour_q[4] += np.deg2rad(115)
    print(f"Robot q after pouring: {np.rad2deg(pour_q)}")
    controller.animate_trajectory(robot2, robot2.q, pour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished Pouring")

    # Step 7: Rotate the drink back up
    print("\n[R2] Un-pouring...")
    unpour_q = robot2.q.copy()
    unpour_q[4] -= np.deg2rad(115)
    controller.animate_trajectory(robot2, robot2.q, unpour_q, steps=60)
    controller.print_pose(robot2, "R2 Finished un-pouring")

    # Step 8: Return the drink
    print("\n[R2] Returning drink to shelf...")
    controller.animate_trajectory(robot2, robot2.q, hover_q_r2, steps=60)
    controller.move_rmrc(robot2, robot2.q, target_r2_pose, 50)
    controller.drop_object(robot2)

    # Step 9: Retract back from the wall
    print("\n[R2] Retracting from shelf...")
    retract_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.2)
    controller.move_rmrc(robot2, robot2.q, retract_pose, 50)
    controller.print_pose(robot2, "R2 Retracted")

    # Step 10: Swing back to above glass
    print("\n[R2] Swinging around to glass...")
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["GLASS_HOVER"], steps=60)
    controller.print_pose(robot2, "R2 near glass")

    # Step 11: Refine position above glass
    print("\n[R2] Moving back to glass...")
    target_r2_pose = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ SE3(-0.5, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ SE3.Ry(pi)
    final_q, success = controller.find_ikine(robot2, target_r2_pose, 
                                             R2_GUESSES["POUR_HOVER"], "z", False, 0.5)

    if not success:
        print("❌ [R2] Unable to find a valid path to the pouring position.")
        return

    controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    controller.print_pose(robot2, "R2 Hovering above glass")

    # Step 11: Move down to glass
    print("\n[R2] Moving down to glass...")
    controller.move_rmrc(robot2, robot2.q, target_r2_pose, 50)
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    controller.pickup_object(robot2, target_glass)

    # Step 11: Move up with glass
    print("\n[R2] Moving up with glass...")
    target_r2_pose = target_r2_pose @ SE3.Tz(-0.2)
    controller.move_rmrc(robot2, robot2.q, target_r2_pose, 50)

    # Step 12: Swing around to hover glass over table
    print("\n[R2] Swinging around with glass...")
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["GLASS_DROP_HOVER"], steps=60)
    controller.print_pose(robot2, "R2 hovering with glass")

    # Step 13: Refine hover position with glass
    print("\n[R1] bringing glass to correct hover position...")
    r2_target = scene.ROBOT_BASE_POSES["R3_MIXERS"] @ SE3(0.6, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ SE3.Ry(pi)
    r2_q_hover, success = controller.find_ikine(robot2, r2_target, 
                                                R2_GUESSES["GLASS_DROP_HOVER"], "z", False, 0.5)
    if not success:
        print("❌ [R1] Failed to find path to workstation.")
        return
    controller.animate_trajectory(robot2, robot2.q, r2_q_hover, steps=60)
    controller.print_pose(robot2, "R2 at hover before placing glass down")

    # Step 14: Place glass down
    controller.move_rmrc(robot2, robot2.q, r2_target, 50)
    controller.drop_object(robot2)
    controller.print_pose(robot2, "R2 after placing glass down")

    #Step 15: Return to Home Position
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["HOME"], steps=60)
    controller.print_pose(robot2, "R2 in safe home position")

def run_sequence3(controller, robot3, robot2, scene):
    """Executes the ingredient adding sequence for Robot 3 (IngredientBot)."""
    print("\n" + "="*70)
    print(">>> ROBOT 3: ADDING INGREDIENT <<<")
    print("="*70 + "\n")

    # Configuration
    CUBE_INDEX = 8
    HOVER_HEIGHT_R3 = 0.1
    DROP_HEIGHT_R3 = 0.2
    PLACEMENT_DEPTH = 0.04
    X_GLASS, Y_GLASS, Z_GLASS = -1.0, -0.575, 1.2
    VERTICAL_ORIENTATION = SE3.Rx(pi)
    X_UR3_MAT, Y_UR3_MAT, Z_UR3_MAT = 0, 0.55, 1 #0.62

    # Target Objects & Poses
    cube_target = scene.cube_objects[CUBE_INDEX]
    PICK_POSE = scene.cube_poses[CUBE_INDEX] @ VERTICAL_ORIENTATION
    HOVER_POSE = PICK_POSE @ SE3.Tz(-HOVER_HEIGHT_R3)
    PLACE_POSE = SE3(X_GLASS, Y_GLASS, Z_GLASS) @ SE3.Tz(PLACEMENT_DEPTH) @ SE3.Ry(pi)
    DROP_POSE = PLACE_POSE @ SE3.Tz(DROP_HEIGHT_R3)
    UR3_MAT_POSE = SE3(X_UR3_MAT, Y_UR3_MAT, Z_UR3_MAT) 
    

    # Step 1: Approach the Chosen Ingredients 
    print("\n[R3] Approaching cube...")
    hover_q_r3, success = controller.find_ikine(robot3, PICK_POSE, R3_GUESSES["PICKUP_YELLOW"], "z", False, 0.5)
    if not success:
        print("❌ [R2] Failed to find path to ingredients.")
        return

    controller.animate_trajectory(robot3, robot3.q, hover_q_r3, steps=60)
    controller.print_pose(robot3, f"R3 at Hover before Cube {CUBE_INDEX}")

    # Step 2: Pick up the object
    controller.pickup_object(robot3, cube_target)

    # Step 3: Retract back to Hover Pose
    print("\n[R3] Retracting to hover...")
    HOVER_R3 = robot3.fkine(robot3.q) @ SE3.Tz(-0.3)
    controller.move_rmrc(robot3, robot3.q, HOVER_R3, 80)
    controller.print_pose(robot3, "R3 Retracted")

    # Step 4: Swing aroud to drop ingredients 
    print("\n[R3] Rotating...")
    print(f"Robot q before turning: {np.rad2deg(robot3.q)}")
    spin_q = robot3.q.copy()
    spin_q[0] += np.deg2rad(170)
    print(f"Robot q after pouring: {np.rad2deg(spin_q)}")
    controller.animate_trajectory(robot3, robot3.q, spin_q, steps=60)
    controller.print_pose(robot3, "R2 Finished Rotating")

    # Step 5: Lower Down to Deposit Ingredients
    print("\n[R3] Depositing Ingredients...")
    drop_q_r3, success = controller.find_ikine(robot3, PLACE_POSE, R3_GUESSES["DEPOSIT_INGREDIENTS"]) #"z", False, 0.5)
    if not success:
        print("❌ [R2] Failed to find path to ingredients.")
        return

    controller.animate_trajectory(robot3, robot3.q, drop_q_r3, steps=80)
    controller.print_pose(robot3, "R3 at Drop Hover")

    # Step 6: Deposit Ingredient
    controller.drop_object(robot3)

    # Step 7: Return to safe home position
    print("\n[R3] Returning home...")
    controller.animate_trajectory(robot3, robot3.q, R3_GUESSES["HOME"], steps=60)
    controller.print_pose(robot3, "R3 in safe home position")

    # Step 8: Robot 2 Moves to complete drink
    print("\n[R2] Drink Handover...")
    r2_target = scene.ROBOT_BASE_POSES["R3_MIXERS"] @ SE3(0.6, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ SE3.Ry(pi)
    r2_q_hover, success = controller.find_ikine(robot2, r2_target, 
                                                R2_GUESSES["GLASS_HANDOVER_PICKUP"])#, "z", False, 0.5)
    if not success:
        print("❌ [R2] Failed to find path to workstation.")
        return
    controller.animate_trajectory(robot2, robot2.q, r2_q_hover, steps=60)
    controller.print_pose(robot2, "R2 Prepare to transport drink")

    # Step 9: Robot 2 picks up complete drink
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    controller.attach_objects(target_glass, cube_target)
    controller.pickup_object(robot2, target_glass)
    
    print("\n[R3] Retracting for handover...")
    HOVER_R2 = robot2.fkine(robot2.q) @ SE3.Tz(-0.3)
    controller.move_rmrc(robot2, robot2.q, HOVER_R2, 80)
    controller.print_pose(robot2, "R2 Retracted")
    

    # Step 10: Swing aroud to drop drink 
    print("\n[R2] Rotating...")
    print(f"Robot q before turning: {np.rad2deg(robot2.q)}")
    spin_q_r2 = robot2.q.copy()
    spin_q_r2[0] -= np.deg2rad(93)
    print(f"Robot q after turning: {np.rad2deg(spin_q_r2)}")
    controller.animate_trajectory(robot2, robot2.q, spin_q_r2, steps=60)
    controller.print_pose(robot2, "R2 Finished Rotating")

    # Step 11: Refine hover position with glass
    print("\n[R1] bringing glass to correct hover position...")
    r2_target_final = UR3_MAT_POSE @ SE3(0, 0, scene.glass_height + scene.BAR_MAT_THICKNESS*2) @ SE3.Ry(pi)
    r2_q_hover_final, success = controller.find_ikine(robot2, r2_target_final, 
                                                R2_GUESSES["GLASS_HANDOVER_HOVER"], "z", False, 0.5)
    if not success:
        print("❌ [R1] Failed to find path to workstation.")
        return
    controller.animate_trajectory(robot2, robot2.q, r2_q_hover_final, steps=60)
    controller.print_pose(robot2, "R2 at hover before placing glass down")

    # Step 12: Place glass down
    controller.move_rmrc(robot2, robot2.q, r2_target_final, 50)
    controller.drop_object(robot2)
    controller.print_pose(robot2, "R2 after placing glass down")

    #Step 13: Return to Home Position
    controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["HOME"], steps=60)
    controller.print_pose(robot2, "R2 in safe home position")








