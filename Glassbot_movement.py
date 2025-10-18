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
TRAJ_STEPS = 60

GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06

# Offset used when attaching mesh to TCP (so glass sits under the TCP)
GLASS_OFFSET = SE3(0, 0, GLASS_HEIGHT/2)
# Orientation so the glass axis matches the gripper orientation used in Rahul's code
GLASS_ROT = SE3.Ry(np.pi/2)

# ----------------------------------------------------
# II. SWIFT OBJECT PARAMETERS
# ----------------------------------------------------

wall_height = 2.5
wall_thickness = 0.05
floor_height = 0.01

table1_length = 4.0
table1_width = 0.75
table1_height = 1.0
table1_offset_from_wall = 0.5
table1_center_y = -1.5 + wall_thickness + table1_offset_from_wall + table1_width / 2

table2_length = 1.5
table2_width = 0.7
table2_height = 1.0
table2_spacing = 1.0
table2_center_y = table1_center_y + (table1_width / 2) + table2_spacing + (table2_width / 2)

glass_table_length = 0.4
glass_table_width = 0.7
glass_table_height = 1.05
glass_table_center_x = table1_length/2 + glass_table_length/2 + 0.01
glass_table_center_y = table1_center_y

# ----------------------------------------------------
# III. ROBOT BASE POSES
# ----------------------------------------------------

ROBOT_BASE_POSES = {
    "R1_ICE_GLASS": SE3(1.8, table1_center_y, table1_height + floor_height),
    "R2_ALCOHOL": SE3(0.0, table1_center_y, table1_height + floor_height),
    "R3_MIXERS": SE3(-1.6, table1_center_y, table1_height + floor_height),
    "R4_SERVER": SE3(0.2, table2_center_y, table2_height + floor_height),
}

# ----------------------------------------------------
# IV. ENVIRONMENT SETUP
# ----------------------------------------------------

env = swift.Swift()
env.launch(realtime=True)

floor = Cuboid(scale=[6, 3.25, 0.02], color=[0.25, 0.3, 0.35, 1], pose=SE3(0, -0.125, floor_height))
env.add(floor)

back_wall = Cuboid(scale=[6, wall_thickness, wall_height], color=[0.85, 0.85, 0.9, 1], pose=SE3(0, -1.75, wall_height/2))
env.add(back_wall)

left_wall = Cuboid(scale=[wall_thickness, 3.25, wall_height], color=[0.85, 0.85, 0.9, 1], pose=SE3(-3, -0.125, wall_height/2))
env.add(left_wall)

right_wall = Cuboid(scale=[wall_thickness, 3.25, wall_height], color=[0.85, 0.85, 0.9, 1], pose=SE3(3, -0.125, wall_height/2))
env.add(right_wall)

# ----------------------------------------------------
# V. TABLES
# ----------------------------------------------------

tables = [
    {"name": "Workstation", "length": table1_length, "width": table1_width, "height": table1_height, "center": SE3(0, table1_center_y, 0)},
    {"name": "UR3e Table", "length": table2_length, "width": table2_width, "height": table2_height, "center": SE3(0, table2_center_y, 0)},
    {"name": "Glass Table", "length": glass_table_length, "width": glass_table_width, "height": glass_table_height, "center": SE3(glass_table_center_x, glass_table_center_y, 0)},
]

for t in tables:
    cx, cy, cz = t["center"].t
    h = t["height"]
    l = t["length"]
    w = t["width"]
    base = Cuboid(scale=[l, w, h-0.05], color=[0.1, 0.1, 0.15, 1], pose=SE3(cx, cy, (h-0.05)/2))
    env.add(base)
    top = Cuboid(scale=[l, w, 0.05], color=[0.0, 0.6, 0.8, 1], pose=SE3(cx, cy, h - 0.025))
    env.add(top)

# ----------------------------------------------------
# VI. ROBOTS
# ----------------------------------------------------

robot1 = Glassbot()
robot1.base = ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

robot2 = IngredientBot()
robot2.q = robot2.home_q
robot2.base = ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

robot3 = Drinkbot()
robot3.q = robot3.home_q
robot3.base = ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

robot4 = Serverbot()
robot4.base = ROBOT_BASE_POSES["R4_SERVER"] * SE3.Rx(pi/2) * SE3.Ry(pi/2)
robot4.add_to_env(env)

# ----------------------------------------------------
# VII. GLASS OBJECTS (3x3 grid as original)
# ----------------------------------------------------

glass_objects = []
glass_color = [1.0, 0.4, 0.0, 0.7]

width_fractions = [0.1, 0.5, 0.9]
length_fractions = [0.1, 0.5, 0.9]

for yf in width_fractions:
    for xf in length_fractions:
        x_pos = glass_table_center_x - glass_table_length/2 + xf * glass_table_length
        y_pos = glass_table_center_y - glass_table_width/2 + yf * glass_table_width
        z_pos = glass_table_height + GLASS_HEIGHT / 2
        g = Cylinder(radius=GLASS_RADIUS, length=GLASS_HEIGHT, color=glass_color, pose=SE3(x_pos, y_pos, z_pos))
        env.add(g)
        glass_objects.append(g)

# pick the centre glass (index 4)
glass_index = 4
target_glass = glass_objects[glass_index]

# ----------------------------------------------------
# VIII. MOVEMENT HELPERS (IK + safer Cartesian place)
# ----------------------------------------------------

def wrap_to_near(q_goal, q_ref):
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi


def solve_ik(robot, T_target, q0, mask=[1,1,1,0,0,0]):
    # try to solve with mask first; then fallback to full 6 dof
    sol = robot.ikine_LM(T_target, q0=q0, mask=mask, joint_limits=True)
    if sol.success:
        return sol
    sol2 = robot.ikine_LM(T_target, q0=q0, mask=[1,1,1,1,1,1], joint_limits=True)
    return sol2


def move_joint_traj(robot, q_start, q_goal, steps=60, carry_mesh=None):
    if q_goal is None:
        return q_start
    q_goal = wrap_to_near(q_goal, q_start)
    for q in rtb.jtraj(q_start, q_goal, steps).q:
        robot.q = q
        # update gripper geometry if present
        try:
            robot._update_fingers()
        except Exception:
            pass
        if carry_mesh is not None:
            T_tcp = robot.fkine(q)
            carry_mesh.T = (T_tcp * GLASS_OFFSET * GLASS_ROT).A
        env.step(SIM_STEP_TIME)
    return q_goal


def cartesian_place_and_release(robot, q_current, place_xy, final_z, tcp_orientation=SE3.Ry(np.pi/2),
                                approach_height=0.15, lift_after=0.08, final_offset_z=0.015):
    # Move above place
    T_approach = SE3(place_xy[0], place_xy[1], final_z + approach_height) * tcp_orientation
    sol_ap = solve_ik(robot, T_approach, q_current)
    if not sol_ap.success:
        print("[Glassbot] ❌ IK failed for place approach")
        return q_current
    q_now = move_joint_traj(robot, q_current, sol_ap.q, steps=50, carry_mesh=target_glass)

    # Descend to place
    T_place = SE3(place_xy[0], place_xy[1], final_z + final_offset_z) * tcp_orientation
    sol_place = solve_ik(robot, T_place, q_now)
    if not sol_place.success:
        print("[Glassbot] ❌ IK failed for place descend")
        return q_now
    q_now = move_joint_traj(robot, q_now, sol_place.q, steps=50, carry_mesh=target_glass)

    # open gripper (visual)
    try:
        robot.gripper_open(steps=20)
    except Exception:
        pass

    # detach mesh: keep it at placed transform
    T_tcp_final = robot.fkine(q_now) * GLASS_OFFSET * GLASS_ROT
    target_glass.T = T_tcp_final.A

    # lift after release
    T_lift = SE3(place_xy[0], place_xy[1], final_z + lift_after) * tcp_orientation
    sol_lift = solve_ik(robot, T_lift, q_now)
    if sol_lift.success:
        q_now = move_joint_traj(robot, q_now, sol_lift.q, steps=40)
    return q_now

# ----------------------------------------------------
# IX. GLASSBOT PICK-AND-PLACE SEQUENCE (1 glass only)
# ----------------------------------------------------

print("\n" + "="*70)
print(">>> GLASSBOT: PICKING 1 GLASS (IK) <<<")
print("="*70 + "\n")

# Start from robot's current joint config
q_now = robot1.q.copy()
robot1._update_fingers()

# Pre-grasp: a little offset above the glass
T_pre_grasp = SE3(target_glass.T[0:3, 3][0], target_glass.T[0:3, 3][1], target_glass.T[0:3, 3][2] + 0.06) * SE3.Ry(np.pi/2)
sol_pre = solve_ik(robot1, T_pre_grasp, q_now)
if not sol_pre.success:
    print("[Glassbot] ❌ IK failed for pre-grasp")
else:
    q_now = move_joint_traj(robot1, q_now, sol_pre.q, steps=50)

# Grasp: descend to pickup (z ~ table + small offset)
T_pick = SE3(target_glass.T[0:3, 3][0], target_glass.T[0:3, 3][1], target_glass.T[0:3, 3][2] + 0.02) * SE3.Ry(np.pi/2)
sol_pick = solve_ik(robot1, T_pick, q_now)
if not sol_pick.success:
    print("[Glassbot] ❌ IK failed for pickup")
else:
    q_now = move_joint_traj(robot1, q_now, sol_pick.q, steps=40)
    # close gripper (visual)
    try:
        robot1.gripper_close(steps=20)
    except Exception:
        pass
    # attach glass to TCP
    T_tcp = robot1.fkine(q_now)
    target_glass.T = (T_tcp * GLASS_OFFSET * GLASS_ROT).A

# Lift clear
T_lift = SE3(target_glass.T[0:3, 3][0], target_glass.T[0:3, 3][1], target_glass.T[0:3, 3][2] + 0.18) * SE3.Ry(np.pi/2)
sol_lift = solve_ik(robot1, T_lift, q_now)
if sol_lift.success:
    q_now = move_joint_traj(robot1, q_now, sol_lift.q, steps=50, carry_mesh=target_glass)
else:
    print("[Glassbot] ❌ IK failed for lift")

# Place location near R2: choose a conservative reachable point in front of R2
place_x = 0.45  # metres (adjust if you want it closer/further)
place_y = table1_center_y
place_z = table1_height + 0.03

q_now = cartesian_place_and_release(robot1, q_now, (place_x, place_y), final_z=place_z, tcp_orientation=SE3.Ry(np.pi/2))

print("✓ Glassbot pick-and-place finished")

env.hold()
