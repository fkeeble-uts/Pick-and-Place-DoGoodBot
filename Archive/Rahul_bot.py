<<<<<<< HEAD:Rahul_bot.py
import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from IRB120 import ABB_IRB120
from spatialgeometry import Cylinder, Cuboid

# ------------------ Setup environment ------------------ #
env = swift.Swift()
env.launch(realtime=True)

robot = ABB_IRB120()
robot.add_to_env(env)

# ------------------ Objects ------------------ #
GLASS_RADIUS = 0.03
GLASS_HEIGHT = 0.06

glass_posex = 0.55
glass_posey = 0.20
glass_pose = SE3(glass_posex, glass_posey, GLASS_HEIGHT/2)
wine_glass = Cylinder(radius=GLASS_RADIUS, length=GLASS_HEIGHT,
                      pose=glass_pose, color=[0.85, 0.65, 0.35, 0.8])
env.add(wine_glass)

DISP_X = 0.5
DISP_Y = 0.25
DISP_Z = 0.25
DISP_SIZE = [DISP_X, DISP_Y, DISP_Z]
dispenser_pose_x = 0.6
dispenser_pose_y = -0.2
dispenser_pose = SE3(dispenser_pose_x, dispenser_pose_y, DISP_Z/2)
ice_dispenser = Cuboid(scale=[DISP_X, DISP_Y, DISP_Z], pose=dispenser_pose, color=[0.2, 0.4, 1.0, 1.0])
env.add(ice_dispenser)

# ------------------ Gripper visuals ------------------ #
FINGER_LENGTH = 0.08
FINGER_THICKNESS = 0.01
FINGER_HEIGHT = 0.08
FINGER_GAP_OPEN = 0.06
FINGER_GAP_CLOSED = 0.055
FINGER_BACK_OFFSET = 0.02
FINGER_Z_OFFSET = -0.03

left_finger = Cuboid([FINGER_LENGTH, FINGER_THICKNESS, FINGER_HEIGHT], color=[0.2, 0.8, 0.2, 1])
right_finger = Cuboid([FINGER_LENGTH, FINGER_THICKNESS, FINGER_HEIGHT], color=[0.2, 0.8, 0.2, 1])
env.add(left_finger)
env.add(right_finger)

_finger_gap = FINGER_GAP_OPEN

def _finger_transform(tcp, side, gap):
    y = gap / 2.0 if side == "left" else -gap / 2.0
    return tcp * SE3(-FINGER_BACK_OFFSET, y, FINGER_Z_OFFSET) * SE3.Ry(np.pi)

def _update_fingers(q=None):
    if q is None:
        q = robot.q
    tcp = robot.fkine(q)
    left_finger.T = _finger_transform(tcp, "left", _finger_gap).A
    right_finger.T = _finger_transform(tcp, "right", _finger_gap).A

def gripper_open(steps=20):
    global _finger_gap
    for g in np.linspace(_finger_gap, FINGER_GAP_OPEN, steps):
        _finger_gap = g
        _update_fingers()
        env.step(0.02)

def gripper_close(move_mesh=True, steps=20):
    global _finger_gap
    start = _finger_gap
    target = FINGER_GAP_CLOSED
    tcp_now = robot.fkine(robot.q)
    left_final = _finger_transform(tcp_now, "left", target).t
    right_final = _finger_transform(tcp_now, "right", target).t
    midpoint_final = (left_final + right_final) / 2.0
    orig_R = wine_glass.T[0:3, 0:3].copy()

    for i, g in enumerate(np.linspace(start, target, steps)):
        _finger_gap = g
        _update_fingers()
        if move_mesh:
            alpha = (i + 1) / steps
            wine_glass.T[0:3, 3] = (1 - alpha) * wine_glass.T[0:3, 3] + alpha * midpoint_final
            wine_glass.T[0:3, 0:3] = orig_R
        env.step(0.02)

_update_fingers()

# ------------------ IK helper ------------------ #
def wrap_to_near(q_goal, q_ref):
    return q_ref + (q_goal - q_ref + np.pi) % (2 * np.pi) - np.pi

def solve_ik(T_target, q0):
    sol = robot.ikine_LM(T_target, q0=q0, mask=[1,1,1,1,1,1], joint_limits=True)
    return sol.q if sol.success else None

# ------------------ Carry transform ------------------ #
GLASS_OFFSET = SE3(0, 0, GLASS_HEIGHT / 2.0)
GLASS_ROT = SE3.Ry(np.pi / 2)

def move_joint_traj(q_start, q_goal, steps=60, carry_mesh=None):
    if q_goal is None:
        return q_start  # Removed the print message
    q_goal = wrap_to_near(q_goal, q_start)
    for q in rtb.jtraj(q_start, q_goal, steps).q:
        robot.q = q
        _update_fingers(q)
        if carry_mesh is not None:
            T_tcp = robot.fkine(q)
            carry_mesh.T = (T_tcp * GLASS_OFFSET * GLASS_ROT).A
        env.step(0.02)
    return q_goal

# ------------------ Smooth Cartesian place + release (fixed) ------------------ #
def cartesian_place_and_release_safe(q_current, tcp_orientation_SE3, place_xy, final_z,
                                     lift_after=0.08, wrist_clearance=0.03, final_offset_z=0.015):
    T_tcp_now = robot.fkine(q_current)
    start_z = T_tcp_now.t[2]
    approach_z = max(start_z, final_z + wrist_clearance + final_offset_z)

    # 1) Move above the placement point
    T_approach = SE3(place_xy[0], place_xy[1], approach_z) * tcp_orientation_SE3
    q_approach = solve_ik(T_approach, q_current)
    if q_approach is not None:
        q_now = move_joint_traj(q_current, q_approach, steps=50, carry_mesh=wine_glass)
    else:
        q_now = q_current

    # 2) Descend to slightly above the final placement (final_offset_z)
    T_place = SE3(place_xy[0], place_xy[1], final_z + final_offset_z) * tcp_orientation_SE3
    q_place = solve_ik(T_place, q_now)
    if q_place is not None:
        q_now = move_joint_traj(q_now, q_place, steps=50, carry_mesh=wine_glass)

    # 3) Open gripper to release
    gripper_open(steps=20)

    # 4) Lift after release
    T_lift = SE3(place_xy[0], place_xy[1], final_z + lift_after) * tcp_orientation_SE3
    q_lift = solve_ik(T_lift, q_now)
    if q_lift is not None:
        q_now = move_joint_traj(q_now, q_lift, steps=30)

    return q_now

# ------------------ Poses ------------------ #
q_home = np.array(robot._qtest).copy()
robot.q = q_home
_update_fingers()

pre_grasp = SE3(glass_posex - 0.02, glass_posey, GLASS_HEIGHT / 2.0) * SE3.Ry(np.pi/2)
pickup_pose = SE3(glass_posex, glass_posey, GLASS_HEIGHT / 2.0) * SE3.Ry(np.pi/2)
lift_pose = SE3(glass_posex, glass_posey, 0.3) * SE3.Ry(np.pi/2)

# ------------------ Safe dispenser waypoints ------------------ #
disp_front_x = dispenser_pose_x - DISP_X / 1.65
clearance = 0.02
push_x = disp_front_x - clearance
safe_z = DISP_Z + 0.05

over_dispenser_pose = SE3(push_x, dispenser_pose_y, safe_z) * SE3.Ry(np.pi/2)
push_pose = SE3(push_x, dispenser_pose_y, DISP_Z / 4) * SE3.Ry(np.pi/2)
touch_pose = push_pose * SE3.Tx(-0.03)

# ------------------ Sequence ------------------ #
q_now = q_home.copy()

# 1) Approach pre-grasp and pick up glass
q_pre = solve_ik(pre_grasp, q_now)
q_now = move_joint_traj(q_now, q_pre)
q_pick = solve_ik(pickup_pose, q_now)
q_now = move_joint_traj(q_now, q_pick)
gripper_close(move_mesh=True, steps=30)

# 2) Lift glass
q_lift = solve_ik(lift_pose, q_now)
q_now = move_joint_traj(q_now, q_lift, carry_mesh=wine_glass)

# 3) Move safely over dispenser (approach)
q_over = solve_ik(over_dispenser_pose, q_now)
q_now = move_joint_traj(q_now, q_over, carry_mesh=wine_glass)

# 4) Descend straight to push height
q_push = solve_ik(push_pose, q_now)
q_now = move_joint_traj(q_now, q_push, carry_mesh=wine_glass)

# 5) Retract slightly
q_touch = solve_ik(touch_pose, q_now)
q_now = move_joint_traj(q_now, q_touch, carry_mesh=wine_glass)

# 6) Return to lift pose safely over dispenser
q_over_back = solve_ik(over_dispenser_pose, q_now)
q_now = move_joint_traj(q_now, q_over_back, carry_mesh=wine_glass)
q_now = move_joint_traj(q_now, q_lift, carry_mesh=wine_glass)

# 7) Return to pick-up area
q_back_pick = solve_ik(pickup_pose, q_now)
q_now = move_joint_traj(q_now, q_back_pick, carry_mesh=wine_glass)

# 8) Smooth place and release (single trajectory)
tcp_orientation = SE3.Ry(np.pi/2)
q_now = cartesian_place_and_release_safe(
    q_now, tcp_orientation,
    (0.15, 0.39),   #Furtherest y position my arm can travel without the glass floating  
    final_z=0.015,
    lift_after=0.08,
    wrist_clearance=0.03,
    final_offset_z=0.015
)

# 9) Return home
q_now = move_joint_traj(q_now, q_home)
env.hold()
