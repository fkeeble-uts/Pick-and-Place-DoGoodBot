# Require libraries
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

import asyncio, threading

_old_init = threading.Thread.__init__

def _new_init(self, *args, **kwargs):
    _old_init(self, *args, **kwargs)
    _old_run = self.run
    def _run_with_loop():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        _old_run()
    self.run = _run_with_loop

threading.Thread.__init__ = _new_init

import swift
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot
import os
from math import pi

point_freq = 1    

# Make a 6DOF planar arm model for CRB15000
l1 = DHLink(d=0.4, a=0, alpha=-pi/2, qlim=[-pi, pi])
l2 = DHLink(d=-0.0863, a=-0.707, alpha=pi, qlim=[-pi, pi])
l3 = DHLink(d=-0.0863, a=-0.110, alpha=-pi/2, qlim=[-pi, pi])
l4 = DHLink(d=0.637, a=0, alpha=-pi/2, qlim=[-pi, pi])
l5 = DHLink(d=0, a=0.08, alpha=-pi/2, qlim=[-pi, pi])
l6 = DHLink(d=0.1, a=0, alpha=0, qlim=[-pi, pi])
robot = DHRobot([l1, l2, l3, l4, l5, l6], name='my_robot')

cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="#3478f6")
robot = cyl_viz.create_cylinders()

workspace = [-3, 3, -3, 3, -0.05, 2]
q = [-pi/2,pi/2,0,0,pi,0]

robot.plot(q, limits=workspace)
input("Enter to close\n")
plt.close()

def on_key(event):
    if event.key == "enter":
        self.stop_event.set()
