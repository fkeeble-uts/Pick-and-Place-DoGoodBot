# robot_models.py
# Some required libraries
import numpy as np
import time
from itertools import combinations
import threading
from typing import List
from spatialmath.base import *
import roboticstoolbox as rtb
from roboticstoolbox import jtraj, Robot, models
from ir_support import RectangularPrism, line_plane_intersection, CylindricalDHRobotPlot

from spatialgeometry import Cylinder, Sphere, Cuboid
from spatialmath import SE3
import swift

# Useful variables
from math import pi

# --- 1. Existing Robot Model (UR3) ---
def get_existing_robot():
    # Using the pre-defined model for one of the available robots
    robot_ur3e = rtb.models.DH.UR3e()
    print("Robot: UR3 loaded.")
    return robot_ur3e

# --- 2. Your New 6-DOF Arm Model (Individual Task) ---
# Start by defining your arm's DH parameters
def get_new_bartender_arm():
    """Defines and returns the model for your new 6-DOF bartender arm."""
    # 6-DOF robot is required
    # You will need to replace these placeholder values with your actual DH parameters
    L = [
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi]), 
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi]),
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi]),
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi]),
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi]),
        rtb.RevoluteDH(d=0, a=0, alpha=0, qlim=[-pi, pi])  
    ]
    
    # Create the robot object
    new_arm = rtb.DHRobot(
        L, 
        name="BartenderArm_A", 
        manufacturer="SafeCo",
        # Optionally set a tool transform
        tool=SE3.Tx(0.1) 
    )
    
    print("New Robot: BartenderArm_A modeled.")
    return new_arm

if __name__ == '__main__':
    # Simple check to see the models
    ur3e = get_existing_robot()
    print("\nUR3e Joints:", ur3e.q)
    
    new_arm = get_new_bartender_arm()
    print("\nNew Arm Joints:", new_arm.q)