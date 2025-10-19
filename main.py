import time
import numpy as np
import swift
from spatialmath import SE3
import roboticstoolbox as rtb
from math import pi

from robot_helpers import RobotController
from DrinkBot import DrinkBot
from IngredientBot import IngredientBot
from GlassBot import GlassBot
from Serverbot import Serverbot
from EnvironmentSetup import Scene
import sequences as seq

# --- SETUP ---
env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([0, 3, 4], [0, 0, 0.5])
scene = Scene(env)
controller = RobotController(env, scene)

# --- ROBOT CREATION & PLACEMENT ---
robot1 = GlassBot()
robot1.base = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"]
robot1.add_to_env(env)

robot2 = DrinkBot()
robot2.q = robot2.home_q
robot2.base = scene.ROBOT_BASE_POSES["R2_ALCOHOL"]
robot2.add_to_env(env)

robot3 = IngredientBot()
robot3.q = robot3.home_q
robot3.base = scene.ROBOT_BASE_POSES["R3_MIXERS"]
robot3.add_to_env(env)

robot4 = Serverbot()
robot4.base = scene.ROBOT_BASE_POSES["R4_SERVER"] 
robot4.q = np.array([0,-pi/2,0,0,0,0])
robot4.add_to_env(env)

# --- RUN SEQUENCES ---
seq.run_robot1_sequence1(controller, robot1, scene)
seq.run_robot2_sequence1(controller, robot2, scene)
seq.run_robot3_sequence1(controller, robot3, scene)

env.hold()