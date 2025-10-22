import time
import numpy as np
from spatialmath import SE3
from math import pi
from SystemState import RobotState, SequenceProgress # Assuming SystemState is available

# ============================================================================
# JOINT CONFIGURATION GUESSES (for IK initialization)
# ============================================================================

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
    "GLASS_HANDOVER_PICKUP": np.deg2rad(np.array([-180, 138.77, -16.77, -0.0, -155.53, -90.43])),
    "GLASS_HANDOVER_HOVER" : np.deg2rad(np.array([89.57, 128.63, -16.52, 0.0, -145.15, -90.43]))
}

R3_GUESSES = {
    "HOME": np.deg2rad(np.array([180., 0., 0., 0., 0., 0.])),
    "PICKUP_YELLOW": np.deg2rad(np.array([0, 47, 73, -32, 91, 0])),
    "PICKUP_GREEN": np.deg2rad(np.array([-20, 47, 65, -30, 89, 0])),
    "PICKUP_BLUE": np.deg2rad(np.array([-34, 53, 65, -30, 89, 0])),
    "DEPOSIT_INGREDIENTS": np.deg2rad(np.array([168.07, 32.05, 68.88, -10.93, 90.0, -101.93])),
}


# ============================================================================
# CHECKPOINT HELPER (SIMPLIFIED FOR DEMO)
# ============================================================================

def check_halt(success, robot_name, progress: SequenceProgress, 
               sequence_id: str, checkpoint_num: int) -> bool:
    """
    Check if motion succeeded and report halt. DOES NOT UPDATE PROGRESS.
    
    Args:
        success: Motion success flag
        robot_name: Name of robot for logging
        progress: SequenceProgress tracker (ignored for this demo)
        sequence_id: Current sequence ID (ignored for this demo)
        checkpoint_num: Current checkpoint number
        
    Returns:
        True if sequence should halt, False to continue
    """
    if not success:
        progress.set_checkpoint(sequence_id, checkpoint_num)
        print(f"  HALTED at {robot_name} checkpoint {checkpoint_num}")
        return True
    
    # Success - simply continue, no checkpoint update
    progress.set_checkpoint(sequence_id, checkpoint_num + 1)
    return False


# ============================================================================
# ROBOT 1: GLASS PICKUP AND PLACEMENT
# ============================================================================

def run_robot1_sequence(controller, robot1, robot2, robot3, robot4, scene, progress: SequenceProgress):
    """
    Robot 1 (GlassBot): Pick up glass and place on workstation.
    Resumable with checkpoints.
    """
    print("\n" + "="*70)
    print(">>> ROBOT 1 (GLASSBOT): GLASS HANDLING <<<")
    print("="*70 + "\n")
    
    SEQUENCE_ID = 'R1'
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    
    # ------------------------------------------------------------------------
    # HOMING (Ensures clean start from TEACH mode)
    # ------------------------------------------------------------------------
        
    def same_position(q1, q2, tol=1e-3):
        return all(abs(a - b) < tol for a, b in zip(q1, q2))

    # Use the predefined R1_GUESSES["HOME"] for consistency
    R1_Q_HOME = R1_GUESSES["HOME"]
    if not same_position(robot1.q, R1_Q_HOME):
        _success, _ = controller.animate_trajectory(robot1, robot1.q, R1_Q_HOME, steps=60)
        if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 0): return

    R2_Q_HOME = R2_GUESSES["HOME"]
    if not same_position(robot2.q, R2_Q_HOME):
        _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_Q_HOME, steps=60)
        if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 0): return

    R3_Q_HOME = R3_GUESSES["HOME"]
    if not same_position(robot3.q, R3_Q_HOME):
            _success, _ = controller.animate_trajectory(robot3, robot3.q, R3_Q_HOME, steps=60)
            if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 0): return
    
    # CHECKPOINT 1: Hover above glass
    print("\n[R1] Moving to hover above glass...")
    r1_target = (scene.glass_poses[glass_index] @ SE3.Tz(scene.glass_height/2) @ SE3.Ry(pi))
    r1_q_hover, success = controller.find_ikine(robot1, r1_target, R1_GUESSES["GLASS_APPROACH"],
    ignore_var="z", ignore_rotation=False, hover_max=0.5)
    if not success:
        print("❌ [R1] Failed to find hover path")
        return
    
    _success, _ = controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 1): return
    
    # CHECKPOINT 2: Grasp glass
    print("\n[R1] Grasping glass...")
    target_grasp = (scene.glass_poses[glass_index] @ 
                    SE3.Tz(scene.glass_height/2) @ SE3.Ry(pi))
    _success, _ = controller.move_rmrc(robot1, target_grasp, 50)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 2): return
    controller.pickup_object(robot1, target_glass)
    
    # CHECKPOINT 3: Lift glass
    print("\n[R1] Lifting glass...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.15)
    _success, _ = controller.move_rmrc(robot1, lift_pose, 50)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 3): return
    
    # CHECKPOINT 4: Move to workstation
    print("\n[R1] Moving to workstation...")
    r1_target = (scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ 
                 SE3(-0.5, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ 
                 SE3.Ry(pi))
    r1_q_hover, success = controller.find_ikine(
        robot1, r1_target, R1_GUESSES["HANDOFF"],
        ignore_var="z", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R1] Failed to find workstation path")
        return
    
    _success, _ = controller.animate_trajectory(robot1, robot1.q, r1_q_hover, steps=60)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 4): return
    
    # CHECKPOINT 5: Place glass
    print("\n[R1] Placing glass...")
    _success, _ = controller.move_rmrc(robot1, r1_target, 50)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 5): return
    controller.drop_object(robot1)

    
    # CHECKPOINT 6: Retract
    print("\n[R1] Retracting...")
    lift_pose = robot1.fkine(robot1.q) @ SE3.Tz(-0.25)
    _success, _ = controller.move_rmrc(robot1, lift_pose, 50)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 6): return
    
    # CHECKPOINT 7: Return home
    print("\n[R1] Returning home...")
    _success, _ = controller.animate_trajectory(robot1, robot1.q, np.zeros(6), steps=60)
    if check_halt(_success, robot1.name, progress, SEQUENCE_ID, 7): return

    
    
    print("✅ Robot 1 sequence complete!")


# ============================================================================
# ROBOT 2: DRINK POURING
# ============================================================================

def run_robot2_sequence(controller, robot2, scene, progress: SequenceProgress):
    """
    Robot 2 (DrinkBot): Fetch drink, pour into glass, return.
    Resumable with checkpoints.
    """
    print("\n" + "="*70)
    print(">>> ROBOT 2 (DRINKBOT): DRINK POURING <<<")
    print("="*70 + "\n")
    
    SEQUENCE_ID = 'R2'
    drink_index = 3
    target_drink = scene.drink_objects[drink_index]
    
    # CHECKPOINT 1: Approach drink
    print("\n[R2] Approaching drink...")
    target_pose = (scene.drink_poses[drink_index] @ 
                    SE3.Ty(scene.drink_radius) @ SE3.Rx(pi/2))
    hover_q, success = controller.find_ikine(
        robot2, target_pose, R2_GUESSES["PICKUP_DRINK"],
        ignore_var="y", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R2] Failed to find drink path")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, hover_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 1): return
    
    # CHECKPOINT 2: Grasp drink
    print("\n[R2] Grasping drink...")
    _success, _ = controller.move_rmrc(robot2, target_pose, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 2): return
    controller.pickup_object(robot2, target_drink)
    
    # CHECKPOINT 3: Retract from shelf
    print("\n[R2] Retracting from shelf...")
    retract_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.2)
    _success, _ = controller.move_rmrc(robot2, retract_pose, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 3): return
    
    # CHECKPOINT 4: Move to pre-pour position
    print("\n[R2] Moving to pour area...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["PRE_POUR_AWAY"], steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 4): return
    
    # CHECKPOINT 5: Move to pour position
    print("\n[R2] Moving to pour position...")
    pour_height = 0.5
    pour_pose = (scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ 
                    SE3(-0.6, 0, scene.glass_height + pour_height) @ 
                    SE3.Rx(pi/2) @ SE3.Ry(pi/2))
    final_q, success = controller.find_ikine(robot2, pour_pose, robot2.q)
    if not success:
        print("❌ [R2] Failed to find pour position")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 5): return
    
    # CHECKPOINT 6: Pour (rotate wrist)
    print("\n[R2] Pouring...")
    pour_q = robot2.q.copy()
    pour_q[4] += np.deg2rad(115)
    _success, _ = controller.animate_trajectory(robot2, robot2.q, pour_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 6): return
    
    # CHECKPOINT 7: Un-pour (rotate back)
    print("\n[R2] Finishing pour...")
    unpour_q = robot2.q.copy()
    unpour_q[4] -= np.deg2rad(115)
    _success, _ = controller.animate_trajectory(robot2, robot2.q, unpour_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 7): return
    
    # CHECKPOINT 8: Return to shelf hover
    print("\n[R2] Returning drink...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, hover_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 8): return
    
    # CHECKPOINT 9: Place drink back
    _success, _ = controller.move_rmrc(robot2, target_pose, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 9): return
    controller.drop_object(robot2)
    
    # CHECKPOINT 10: Retract from shelf
    print("\n[R2] Retracting...")
    retract_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.2)
    _success, _ = controller.move_rmrc(robot2, retract_pose, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 10): return
    
    # CHECKPOINT 11: Move to glass hover
    print("\n[R2] Moving to glass...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["GLASS_HOVER"], steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 11): return
    
    # CHECKPOINT 12: Refine position above glass
    print("\n[R2] Positioning above glass...")
    glass_target = (scene.ROBOT_BASE_POSES["R1_ICE_GLASS"] @ 
                    SE3(-0.5, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ 
                    SE3.Ry(pi))
    final_q, success = controller.find_ikine(
        robot2, glass_target, robot2.q, 
        ignore_var="z", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R2] Failed to find glass position")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 12): return
    
    # CHECKPOINT 13: Pick up glass
    print("\n[R2] Picking up glass...")
    _success, _ = controller.move_rmrc(robot2, glass_target, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 13): return
    
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    controller.pickup_object(robot2, target_glass)
    
    # CHECKPOINT 14: Lift glass
    print("\n[R2] Lifting glass...")
    lift_target = glass_target @ SE3.Tz(-0.2)
    _success, _ = controller.move_rmrc(robot2, lift_target, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 14): return
    
    # CHECKPOINT 15: Swing to drop hover
    print("\n[R2] Moving to drop position...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["GLASS_DROP_HOVER"], steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 15): return
    
    # CHECKPOINT 16: Refine drop hover
    print("\n[R2] Positioning for handoff...")
    drop_target = (scene.ROBOT_BASE_POSES["R3_MIXERS"] @ 
                    SE3(0.6, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ 
                    SE3.Ry(pi))
    drop_q, success = controller.find_ikine(
        robot2, drop_target, robot2.q,
        ignore_var="z", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R2] Failed to find drop position")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, drop_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 16): return
    
    # CHECKPOINT 17: Place glass down
    print("\n[R2] Placing glass...")
    _success, _ = controller.move_rmrc(robot2, drop_target, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 17): return
    controller.drop_object(robot2)
    
    # CHECKPOINT 18: Return home
    print("\n[R2] Returning home...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["HOME"], steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 18): return
    
    print("✅ Robot 2 sequence complete!")


# ============================================================================
# ROBOT 3: INGREDIENT ADDITION
# ============================================================================

def run_robot3_sequence(controller, robot3, robot2, scene, progress: SequenceProgress):
    """
    Robot 3 (IngredientBot): Add ingredient to glass and handoff.
    Resumable with checkpoints.
    """
    print("\n" + "="*70)
    print(">>> ROBOT 3 (INGREDIENTBOT): ADDING INGREDIENTS <<<")
    print("="*70 + "\n")
    
    SEQUENCE_ID = 'R3'
    CUBE_INDEX = 8
    cube_target = scene.cube_objects[CUBE_INDEX]
    
    # CHECKPOINT 1: Approach ingredient
    print("\n[R3] Approaching ingredient...")
    pick_pose = scene.cube_poses[CUBE_INDEX] @ SE3.Rx(pi)
    hover_q, success = controller.find_ikine(
        robot3, pick_pose, R3_GUESSES["PICKUP_YELLOW"],
        ignore_var="z", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R3] Failed to find ingredient path")
        return
    
    _success, _ = controller.animate_trajectory(robot3, robot3.q, hover_q, steps=60)
    if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 1): return
    
    controller.pickup_object(robot3, cube_target)
    
    # CHECKPOINT 2: Retract with ingredient
    print("\n[R3] Retracting...")
    retract_pose = robot3.fkine(robot3.q) @ SE3.Tz(-0.3)
    _success, _ = controller.move_rmrc(robot3, retract_pose, 80)
    if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 2): return
    
    # CHECKPOINT 3: Rotate to glass
    print("\n[R3] Rotating to glass...")
    spin_q = robot3.q.copy()
    spin_q[0] += np.deg2rad(170)
    _success, _ = controller.animate_trajectory(robot3, robot3.q, spin_q, steps=60)
    if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 3): return
    
    # CHECKPOINT 4: Move to deposit position
    print("\n[R3] Moving to deposit position...")
    place_pose = SE3(-1.0, -0.575, 1.2) @ SE3.Tz(0.04) @ SE3.Ry(pi)
    drop_q, success = controller.find_ikine(robot3, place_pose, robot3.q)
    if not success:
        print("❌ [R3] Failed to find deposit position")
        return
    
    _success, _ = controller.animate_trajectory(robot3, robot3.q, drop_q, steps=80)
    if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 4): return
    
    controller.drop_object(robot3)
    
    # CHECKPOINT 5: Return home
    print("\n[R3] Returning home...")
    _success, _ = controller.animate_trajectory(robot3, robot3.q, R3_GUESSES["HOME"], steps=60)
    if check_halt(_success, robot3.name, progress, SEQUENCE_ID, 5): return
    
    print("✅ Robot 3 sequence complete!")
    
    # ========================================================================
    # ROBOT 2 HANDOFF (Part of R3 sequence)
    # ========================================================================
    
    print("\n[R2] Starting handoff sequence...")
    
    # CHECKPOINT 6: R2 moves to glass
    print("\n[R2] Moving to pick up completed drink...")
    handoff_target = (scene.ROBOT_BASE_POSES["R3_MIXERS"] @ 
                      SE3(0.6, 0, scene.glass_height + scene.BAR_MAT_THICKNESS) @ 
                      SE3.Ry(pi))
    handoff_q, success = controller.find_ikine(robot2, handoff_target, robot2.q)
    if not success:
        print("❌ [R2] Failed to find handoff position")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, handoff_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 6): return
    
    # CHECKPOINT 7: R2 picks up glass with ingredient
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    controller.attach_objects(target_glass, cube_target)
    controller.pickup_object(robot2, target_glass)
    
    print("\n[R2] Lifting completed drink...")
    lift_pose = robot2.fkine(robot2.q) @ SE3.Tz(-0.3)
    _success, _ = controller.move_rmrc(robot2, lift_pose, 80)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 7): return
    
    # CHECKPOINT 8: R2 rotates toward serving area
    print("\n[R2] Moving to serving area...")
    spin_q = robot2.q.copy()
    spin_q[0] -= np.deg2rad(93)
    _success, _ = controller.animate_trajectory(robot2, robot2.q, spin_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 8): return
    
    # CHECKPOINT 9: R2 moves to final drop hover
    print("\n[R2] Positioning for final placement...")
    final_target = (SE3(0, 0.55, 1) @ 
                    SE3(0, 0, scene.glass_height + scene.BAR_MAT_THICKNESS * 2) @ 
                    SE3.Ry(pi))
    final_q, success = controller.find_ikine(
        robot2, final_target, robot2.q,
        ignore_var="z", ignore_rotation=False, hover_max=0.5
    )
    if not success:
        print("❌ [R2] Failed to find final position")
        return
    
    _success, _ = controller.animate_trajectory(robot2, robot2.q, final_q, steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 9): return
    
    # CHECKPOINT 10: R2 places completed drink
    print("\n[R2] Placing completed drink...")
    _success, _ = controller.move_rmrc(robot2, final_target, 50)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 10): return
    
    controller.drop_object(robot2)
    
    # CHECKPOINT 11: R2 returns home
    print("\n[R2] Returning home...")
    _success, _ = controller.animate_trajectory(robot2, robot2.q, R2_GUESSES["HOME"], steps=60)
    if check_halt(_success, robot2.name, progress, SEQUENCE_ID, 11): return
    
    print("✅ Sequence 3 finished!")

def run_robot4_sequence(controller, robot4, scene, progress: SequenceProgress):
    """
    Robot 4 (ServeBot): Fetch drink, place in customer collection area
    """
    print("\n" + "="*70)
    print(">>> ROBOT 4 (SERVEBOT): DRINK SERVING <<<")
    print("="*70 + "\n")
    
    SEQUENCE_ID = 3
    CUBE_INDEX = 8
    cube_target = scene.cube_objects[CUBE_INDEX]

    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    
    # Step 1: R4 moves to finished glass
    print("\n[R4] Moving to pick up completed glass...")
    mat_c_x = scene.BAR_MAT_POSITIONS[2]["x"]
    mat_c_y = scene.BAR_MAT_POSITIONS[2]["y"]
    mat_c_z = scene.BAR_MAT_Z_POS
    mat_c_pose = SE3(mat_c_x, mat_c_y, mat_c_z)

    # Now calculate the handoff target relative to the mat pose
    handoff_target = mat_c_pose @ SE3(0, 0, scene.glass_height + scene.BAR_MAT_THICKNESS*2) @ SE3.Ry(pi)
    handoff_q, success = controller.find_ikine(robot4, handoff_target, robot4.q, "z", False, 0.5)
    if not success:
        print("❌ [R4] Failed to hover over finished glass")
        return
    
    _success, _ = controller.animate_trajectory(robot4, robot4.q, handoff_q, steps=60)
    if check_halt(_success, robot4.name, progress, SEQUENCE_ID, 6): return

    _success, _ = controller.move_rmrc(robot4, handoff_target, 80)
    if check_halt(_success, robot4.name, progress, SEQUENCE_ID, 7): return
    
    # Step 2: R4 picks up glass with ingredient
    glass_index = 3
    target_glass = scene.glass_objects[glass_index]
    controller.attach_objects(target_glass, cube_target)
    controller.pickup_object(robot4, target_glass)
    
    print("\n[R4] Lifting completed drink...")
    lift_pose = robot4.fkine(robot4.q) @ SE3.Tz(-0.3)
    _success, _ = controller.move_rmrc(robot4, lift_pose, 80)
    if check_halt(_success, robot4.name, progress, SEQUENCE_ID, 7): return

    # Step 3: Turn to customer collection area
    target_pedestal = scene.ROBOT_BASE_POSES["R4_SERVER"] @ SE3.Tx(-0.5) @ SE3.Tz(scene.glass_height) @ SE3.Ry(pi)
    
    serve_q, success = controller.find_ikine(robot4, target_pedestal, robot4.q, "z", False, 0.5)
    if not success:
        print("❌ [R4] Failed to hover over delivery area")
        return
    
    _success, _ = controller.animate_trajectory(robot4, robot4.q, serve_q, steps=60)
    controller.print_pose(robot4, "Robot 4 hovering over collection point")
    if check_halt(_success, robot4.name, progress, SEQUENCE_ID, 6): return

    # Step 4: Place finished glass down on collection area
    _success, _ = controller.move_rmrc(robot4, target_pedestal, 80)
    if check_halt(_success, robot4.name, progress, SEQUENCE_ID, 7): return
    
