import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from spatialgeometry import Cylinder, Cuboid
from math import pi

from Drinkbot import Drinkbot
from IngredientBot import IngredientBot
from Glassbot import Glassbot
from Serverbot import Serverbot

# ----------------------------------------------------
# I. CONSTANTS & CONFIGURATION
# ----------------------------------------------------
SIM_STEP_TIME = 0.02
GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06
GLASS_OFFSET = SE3(0, 0, GLASS_HEIGHT / 2)
SIDE_GRASP_ROT = SE3.Ry(np.pi / 2)

# ----------------------------------------------------
# II. ENVIRONMENT SETUP
# ----------------------------------------------------
env = swift.Swift()
env.launch(realtime=True)

# Floor and wall
floor = Cuboid(scale=[6, 3.25, 0.02], color=[0.25, 0.3, 0.35, 1], pose=SE3(0, -0.125, 0.01))
env.add(floor)

wall_height = 2.5
wall_thickness = 0.05
back_wall = Cuboid(scale=[6, wall_thickness, wall_height],
                   color=[0.85, 0.85, 0.9, 1],
                   pose=SE3(0, -1.75, wall_height / 2))
env.add(back_wall)

# Tables
table1_height = 1.0
table1_length = 4.0
table1_width = 0.75
table1_center_y = -1.5 + wall_thickness + 0.5 + table1_width / 2

table2_height = 1.0
table2_length = 1.5
table2_width = 0.7
table2_spacing = 1.0
table2_center_y = table1_center_y + (table1_width / 2) + table2_spacing + (table2_width / 2)

glass_table_height = 1.05
glass_table_length = 0.4
glass_table_width = 0.7
glass_table_center_x = table1_length / 2 + glass_table_length / 2 + 0.01
glass_table_center_y = table1_center_y

for (L, W, H, C) in [
    (table1_length, table1_width, table1_height, SE3(0, table1_center_y, 0)),
    (table2_length, table2_width, table2_height, SE3(0, table2_center_y, 0)),
    (glass_table_length, glass_table_width, glass_table_height, SE3(glass_table_center_x, glass_table_center_y, 0)),
]:
    env.add(Cuboid(scale=[L, W, H - 0.05], color=[0.1, 0.1, 0.15, 1], pose=SE3(C.t[0], C.t[1], (H - 0.05) / 2)))
    env.add(Cuboid(scale=[L, W, 0.05], color=[0.0, 0.6, 0.8, 1], pose=SE3(C.t[0], C.t[1], H - 0.025)))

# ----------------------------------------------------
# III. ROBOT BASE POSES
# ----------------------------------------------------
ROBOT_BASE_POSES = {
    "R1_ICE_GLASS": SE3(1.6, table1_center_y, table1_height),
    "R2_ALCOHOL": SE3(0.0, table1_center_y, table1_height),
    "R3_MIXERS": SE3(-1.6, table1_center_y, table1_height),
    "R4_SERVER": SE3(0.2, table2_center_y, table2_height),
}

# ----------------------------------------------------
# IV. LOAD ROBOTS
# ----------------------------------------------------
robot1 = Glassbot()
robot1.base = ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)
home_q = robot1.q.copy()

robot2 = IngredientBot()
robot2.base = ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

robot3 = Drinkbot()
robot3.base = ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

robot4 = Serverbot()
robot4.base = ROBOT_BASE_POSES["R4_SERVER"] * SE3.Rx(pi / 2) * SE3.Ry(pi / 2)
robot4.add_to_env(env)

# ----------------------------------------------------
# V. GLASS OBJECTS
# ----------------------------------------------------
glass_objects = []
glass_color = [1.0, 0.4, 0.0, 0.7]

for yf in [0.1, 0.5, 0.9]:
    for xf in [0.1, 0.5, 0.9]:
        x_pos = glass_table_center_x - glass_table_length / 2 + xf * glass_table_length
        y_pos = glass_table_center_y - glass_table_width / 2 + yf * glass_table_width
        z_pos = glass_table_height + GLASS_HEIGHT / 2
        g = Cylinder(radius=GLASS_RADIUS, length=GLASS_HEIGHT,
                     color=glass_color, pose=SE3(x_pos, y_pos, z_pos))
        env.add(g)
        glass_objects.append(g)

# Find closest glass to Glassbot
base_pos = ROBOT_BASE_POSES["R1_ICE_GLASS"].t[0:2]
target_glass = glass_objects[int(np.argmin([np.linalg.norm(g.T[0:2, 3] - base_pos) for g in glass_objects]))]

# ----------------------------------------------------
# VI. HELPER FUNCTIONS
# ----------------------------------------------------
def wrap_to_near(q_goal, q_ref):
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi

def solve_ik(robot, T_target, q0):
    sol = robot.ikine_LM(T_target, q0=q0, mask=[1]*6, joint_limits=True)
    if not sol.success:
        print(f"❌ IK failed for target:\n{T_target}")
        return None
    return sol.q

def move_joint_traj(robot, q_start, q_goal, steps=60, carry_mesh=None):
    if q_goal is None:
        print("⚠️ Skipping motion: IK failed.")
        return q_start
    q_goal = wrap_to_near(q_goal, q_start)
    for q in rtb.jtraj(q_start, q_goal, steps).q:
        robot.q = q
        try:
            robot._update_fingers()
        except Exception:
            pass
        if carry_mesh is not None:
            T_tcp = robot.fkine(q)
            carry_mesh.T = (T_tcp * GLASS_OFFSET * SIDE_GRASP_ROT).A
        env.step(SIM_STEP_TIME)
    return q_goal

# ----------------------------------------------------
# VII. PICK AND PLACE SEQUENCE
# ----------------------------------------------------
print("\n>>> Glassbot pick, move in x, and place at x=1.2 <<<\n")

q_now = home_q.copy()
robot1._update_fingers()
g_pos = target_glass.T[0:3, 3]

# 1. Pre-grasp
T_pre = SE3(g_pos[0] - 0.08, g_pos[1], g_pos[2] + 0.05) * SIDE_GRASP_ROT
q_pre = solve_ik(robot1, T_pre, q_now)
q_now = move_joint_traj(robot1, q_now, q_pre, steps=40)

# 2. Pick
T_pick = SE3(g_pos[0] - 0.015, g_pos[1], g_pos[2]) * SIDE_GRASP_ROT
q_pick = solve_ik(robot1, T_pick, q_now)
q_now = move_joint_traj(robot1, q_now, q_pick, steps=40, carry_mesh=target_glass)
robot1.gripper_close(steps=25)

# 3. Lift
T_lift = SE3(g_pos[0] - 0.015, g_pos[1], g_pos[2] + 0.25) * SIDE_GRASP_ROT
q_lift = solve_ik(robot1, T_lift, q_now)
q_now = move_joint_traj(robot1, q_now, q_lift, steps=50, carry_mesh=target_glass)

# 4. Move to x = 1.2 (within reachable range)
target_x = 1.2
T_move = SE3(target_x, g_pos[1], g_pos[2] + 0.25) * SIDE_GRASP_ROT
q_move = solve_ik(robot1, T_move, q_now)
q_now = move_joint_traj(robot1, q_now, q_move, steps=60, carry_mesh=target_glass)

# 5. Place down
T_place = SE3(target_x, g_pos[1], g_pos[2]) * SIDE_GRASP_ROT
q_place = solve_ik(robot1, T_place, q_now)
q_now = move_joint_traj(robot1, q_now, q_place, steps=50, carry_mesh=target_glass)
robot1.gripper_open(steps=20)

# 6. Lift up and return home
T_lift_after = SE3(target_x, g_pos[1], g_pos[2] + 0.25) * SIDE_GRASP_ROT
q_lift_after = solve_ik(robot1, T_lift_after, q_now)
q_now = move_joint_traj(robot1, q_now, q_lift_after, steps=50)
q_now = move_joint_traj(robot1, q_now, home_q, steps=70)

print("✓ Glass successfully picked, moved, and placed at x=1.2.\n")

env.hold()
