import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid, Box
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
# HELPER FUNCTiONS
# ============================================================================

def wrap_to_near(q_goal, q_ref):
    """Wrap joint angles to nearest equivalent to reference."""
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi

def move_to_q(robot, q_target, steps=scene.TRAJ_STEPS, name="", carry_object=None):
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
        
        env.step(scene.SIM_STEP_TIME)
    
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
# SAVED JOINT POSES FROM TEACH MODE
# ============================================================================

# R1 (Glassbot)
R1_POSES = {
    "HOME": np.deg2rad(np.array([0., 0., 0., 0., 0., 0.])),
    "GLASS_APPROACH": np.deg2rad(np.array([0., 40.68, 16.82, 9.09, 0., 0.])),
    "GLASS_PICKUP": np.deg2rad(np.array([0., 62.27, 0.45, 10.91, 0., 0.])),
    "LIFT_CLEAR": np.deg2rad(np.array([0., 39.55, -11.82, 10.91, 0., 0.])),
    "ICE_MACHINE": np.deg2rad(np.array([-87.95, 52.05, 3.52, 10.91, 0., 0.])),
    "HANDOFF": np.deg2rad(np.array([-180., 92.95, -61.93, 10.91, 0., 0.])),
}

# R2 (Drinkbot)
R2_POSES = {
    "HOME": robot2.q.copy(),
    "PICKUP_DRINK": np.deg2rad(np.array([-74.207, 141.295, -31.751, 9.875, 103.964, -24.255])),  # Position to pick up drink
    "PLACE_GLASS": np.deg2rad(np.array([0, 25.495, 174.686, 0, -208, 0]))    # Position to place drink
}

# R3 (Drinkbot) 
R3_POSES = {
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

glass_index = 4  # Use middle glass
target_glass = scene.glass_objects[glass_index]
held_by_r1 = False
held_by_r2 = False

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
# ROBOT 2 SEQUENCE - ADD ALCOHOL TO GLASS
# ============================================================================

print("\n" + "="*70)
print(">>> ROBOT 2: MOVING TO DRINK 4 <<<")
print("="*70 + "\n")

time.sleep(1.0)

print("[R2] Moving to drinks shelf...")
print("Target pose of drink 3:")
print(scene.drink_poses[3])
print("Drinkbot pose:")
print(scene.ROBOT_BASE_POSES["R2_ALCOHOL"])
print("Relative pose from drinkbot to drink 4:")
print(scene.drink_poses[3]-scene.ROBOT_BASE_POSES["R2_ALCOHOL"])

# Step 7: Move to drink 4
print("\n[R2] Moving to drink 4...")
q_now_r2 = move_to_q(robot2, R2_POSES["PICKUP_DRINK"], steps=60, name="Drink4",
                      carry_object=None)
print_pose(robot1, "R2 at Drink 4")
time.sleep(0.5)

env.hold()