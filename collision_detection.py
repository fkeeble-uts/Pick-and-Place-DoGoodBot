# Require libraries
from ast import List
import numpy as np
import threading
from itertools import combinations
import swift
import time
from spatialmath.base import *
from spatialgeometry import Cuboid, Sphere
from typing import List
from roboticstoolbox import DHRobot, DHLink, jtraj
from ir_support import RectangularPrism, line_plane_intersection, CylindricalDHRobotPlot

# Useful variables
from math import pi, radians

# ---------------------------------------------------------------------------------------#
class Lab5Solution:
    def __init__(self):
        print("Lab 5 Solution - Questions 2 + 3")
        self.stop_event = threading.Event()         # Event to end teach mode when 'enter' pressed


    def wait_for_enter(self):
        '''
        Helper threaded function to detect keypress without needing keyboard library
        '''
        try:
            #print("Press Enter to stop.\n")
            input()
        except EOFError:
            pass
        self.stop_event.set()


    def question2_and_3(self):
        # 2.1) Make the 3DOF planar arm model
        l1 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
        l2 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
        l3 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
        robot = DHRobot([l1, l2, l3], name='my_robot')
        robot.q = [-pi/2, 0, 0]   # Define initial joint state for robot

        # Creating simple cylindrical geometry for each link of the DHRobot (but this time all links will be blue)
        cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="#3478f6")
        robot = cyl_viz.create_cylinders()

        # Create Swift environment and add robot
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(robot)

        # 2.2/2.3)
        lwh = [1.5, 1.5, 1.5]       # Python list containing the desired length (X), width (Y), height (Z) of the cuboid object
        centre = [2, 0, -0.5]       # Python list containing XYZ centre of cuboid
        pose = transl(centre)       # Define the pose of the centre of the cuboid
        # Create prism/cuboid and set desired pose
        prism = Cuboid(scale=lwh, color=[0.0, 1.0, 0.0, 0.5])   # Set colour to green, but with some transparency to see through it (RGBA)
        prism.T = pose
        env.add(prism)              # Add prism to environment

        # Get the prism's mesh properties (vertices, faces, face normals) using RectangularPrism class from ir_support
        # Note: RectangularPrism first 3 parameters are length (X), width (Y), height (Z)
        vertices, faces, face_normals = RectangularPrism(lwh[0], lwh[1], lwh[2], center=centre).get_data()

        env.set_camera_pose([1, 2, 2], [1.5, 0, 0])  # Set camera pose (position, look-at)
        input("Press enter to continue, and then enter again to exit teach mode\n")

        # Create Teach Mode in Swift
        # 1. Calculate XYZ, Roll, Pitch, Yaw of robot end-effector
        xyz, rpy = self.calculate_dof(robot)

        # 2. Creating Labels to show XYZ, Roll, Pitch, Yaw
        # XYZ
        text_x, text_y, text_z = f"X: {xyz[0]}", f"Y: {xyz[1]}", f"Z: {xyz[2]}"
        label_x, label_y, label_z = swift.Label(text_x), swift.Label(text_y), swift.Label(text_z)
        # RPY
        text_roll, text_pitch, text_yaw = f"Roll (φ): {rpy[0]}", f"Pitch (θ): {rpy[1]}", f"Yaw (ψ): {rpy[2]}"
        label_roll, label_pitch, label_yaw = swift.Label(text_roll), swift.Label(text_pitch), swift.Label(text_yaw)
        # Create list of labels
        labels = [label_x, label_y, label_z, label_roll, label_pitch, label_yaw]

        # 3. Creating Sliders to control joint angles
        # Note: We use lambda functions so that we don't have to create a different callback function for each slider/joint
        # Therefore, no matter the number of joints on the robot, this setup will work as long as the index in the callback parameter:
        # self.slider_callback(value, index, robot) corresponds to joint being controlled by that slider (zero-index - starting with joint 0)
        # 'Value' is the slider value which is returned by Swift when the callback is run - we then use that value as a parameter to update the joint
        slider_joint1 = swift.Slider(cb=lambda value: self.slider_callback(value, 0, robot, labels), min=-180, max=180, step=1, value=-90, desc='Joint 1', unit='°')
        slider_joint2 = swift.Slider(cb=lambda value: self.slider_callback(value, 1, robot, labels), min=-180, max=180, step=1, value=0, desc='Joint 2', unit='°')
        slider_joint3 = swift.Slider(cb=lambda value: self.slider_callback(value, 2, robot, labels), min=-180, max=180, step=1, value=0, desc='Joint 3', unit='°')
        
        # 4. Adding Labels and Sliders to environment
        # Labels
        env.add(label_x)
        env.add(label_y)
        env.add(label_z)
        env.add(label_roll)
        env.add(label_pitch)
        env.add(label_yaw)
        # Sliders
        env.add(slider_joint1)
        env.add(slider_joint2)
        env.add(slider_joint3)


        # Continuously update the teach figure while it is being used (every 0.05s)
        input_thread = threading.Thread(target=self.wait_for_enter)
        input_thread.start()

        collisions = []      # List tracking Sphere's made in swift representing collisions with robot and object
        while not self.stop_event.is_set():
        # 2.4) Get the transform of every joint (i.e. start and end of every link)
            tr = get_link_poses(robot)

        # 2.5) Go through each link and also each triangle face
            for i in range(np.size(tr,2)-1):
                for j, face in enumerate(faces):
                    vert_on_plane = vertices[face][0]
                    intersect_p, check = line_plane_intersection(face_normals[j], 
                                                                vert_on_plane, 
                                                                tr[i][:3,3], 
                                                                tr[i+1][:3,3])
                    # list of all triangle combination in a face
                    triangle_list  = np.array(list(combinations(face,3)),dtype= int)
                    if check == 1:
                        for triangle in triangle_list:
                            if is_intersection_point_inside_triangle(intersect_p, vertices[triangle]):
                                # Create a red sphere in Swift at the intersection point - if lagging, reduce radius
                                new_collision = Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
                                new_collision.T = transl(intersect_p[0], intersect_p[1], intersect_p[2])
                                env.add(new_collision)
                                collisions.append(new_collision)
                                #print('Intersection')
                                break

            env.step()
            time.sleep(0.05)
        input_thread.join()

        # Reset the Swift environment to remove spheres and teach sliders - have to re-add robot and prism
        env.reset()
        env.add(robot)
        env.add(prism)
        env.set_camera_pose([-1.5, 0, 2], [2, 0, 0])  # (position, look-at)
        collisions.clear()      # Clear the list holding previous collision sphere objects

        # 2.6) Go through until there are no step sizes larger than 1 degree
        q1 = [-pi/4,0,0]
        q2 = [pi/4,0,0]
        steps = 2
        while np.any(1 < np.abs(np.diff(np.rad2deg(jtraj(q1,q2,steps).q), axis= 0))):
            steps+=1
        q_matrix = jtraj(q1,q2,steps).q

        # 2.7
        print("Question 2.7 - Press enter to continue\n")
        result = [True for _ in range(steps)]
        for i,q in enumerate(q_matrix):
            robot.q = q
            result[i] = is_collision(robot, [q], faces, vertices, face_normals, collisions, env=env, return_once_found=False)
            env.step()
            time.sleep(0.05)
        input("Enter to continue to question 3.1\n")


        ## Question 3
        # Reset environment to remove collision spheres due to NoneType bug in .remove() and .step()
        env.reset()
        env.add(robot)
        env.add(prism)
        env.set_camera_pose([-1.5, 0, 2], [2, 0, 0])  # (position, look-at)
        collisions.clear()      # Clear the list holding previous collision sphere objects

        # 3.1) Add some waypoints to q_matrix
        robot.q = q1
        env.step()
        time.sleep(0.01)
        q_waypoints =[q1, 
                    [radians(x) for x in [-45, -111, 72]],
                    [radians(x) for x in [169, -111, -72]],
                    q2]
        q_matrix = interpolate_waypoints_radians(q_waypoints, np.deg2rad(5))

        if is_collision(robot, q_matrix, faces, vertices, face_normals, collisions, return_once_found=False):
            print('Collision detected!')
        else:
            print('No collision found')

        for q in q_matrix:
            robot.q = q
            result = is_collision(robot, [q], faces, vertices, face_normals, collisions, env=env, return_once_found=False)
            env.step()
            time.sleep(0.05)

        input('Enter to continue to question 3.2\n')

        # Reset environment to remove collision spheres due to NoneType bug in .remove() and .step()
        env.reset()
        env.add(robot)
        env.add(prism)
        env.set_camera_pose([-1, -1.5, 2], [2, 0, 0])  # (position, look-at)
        collisions.clear()      # Clear the list holding previous collision sphere objects

        # 3.2) Manually create cartesian waypoints
        q_waypoints = [q1]
        q_waypoints.append(robot.ikine_LM(transl(1.5,-1,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1,-1,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1.1,-0.5,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1.1,0,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1.1,0.5,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1.1,1,0), q0= q_waypoints[-1], mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(robot.ikine_LM(transl(1.5,1,0), q0= q2, mask= [1,1,0,0,0,0]).q)
        q_waypoints.append(q2)
        q_matrix = interpolate_waypoints_radians(q_waypoints, np.deg2rad(5))
        
        if is_collision(robot, q_matrix, faces, vertices, face_normals, collisions, return_once_found=False):
            print('Collision detected!')
        else:
            print('No collision found')

        for q in q_matrix:
            robot.q = q
            result = is_collision(robot, [q], faces, vertices, face_normals, collisions, env=env, return_once_found=False)
            env.step()
            time.sleep(0.05)

        input('Enter to continue to question 3.3\n')

         # Reset environment to remove collision spheres due to NoneType bug in .remove() and .step()
        env.reset()
        env.add(robot)
        env.add(prism)
        env.set_camera_pose([-2, 0, 2.5], [1, 0, 0])  # (position, look-at)
        collisions.clear()      # Clear the list holding previous collision sphere objects

        # 3.3) Randomly select waypoints (primative RRT)
        robot.q = q1
        env.step()
        time.sleep(0.01)
        q_waypoints = np.array([q1, q2])
        is_collision_check = True
        checked_till_waypoint = 0
        q_matrix = []
        while is_collision_check:
            start_waypoint = checked_till_waypoint
            for i in range(start_waypoint, len(q_waypoints)-1):
                q_matrix_join = interpolate_waypoints_radians([q_waypoints[i], q_waypoints[i+1]], np.deg2rad(10))

                if not is_collision(robot, q_matrix_join, faces, vertices, face_normals, collisions, return_once_found=True):
                    q_matrix.extend(q_matrix_join)
                    for q in q_matrix_join:
                        robot.q = q
                        env.step()
                        time.sleep(0.01)
                    is_collision_check = False
                    checked_till_waypoint = i+1

                    # Now try to join to the final goal q2
                    q_matrix_join = interpolate_waypoints_radians([q_matrix[-1], q2], np.deg2rad(10))
                    if not is_collision(robot, q_matrix_join, faces, vertices, face_normals, collisions, return_once_found=True):
                        q_matrix.extend(q_matrix_join)
                        for q in q_matrix_join:
                            robot.q =q
                            env.step()
                            time.sleep(0.01)
                        # Reached goal without collision, so break out
                        break
                else:
                    # Randomly pick a pose that is not in collision
                    q_rand = (2 * np.random.rand(1, 3) - 1) * np.pi
                    q_rand = q_rand.tolist()[0]  # Convert to a 3-element list

                    while is_collision(robot, [q_rand], faces, vertices, face_normals, collisions, return_once_found=True):
                        q_rand = (2 * np.random.rand(1, 3) - 1) * np.pi
                        q_rand = q_rand.tolist()[0]  # Convert to a 3-element list
                    q_waypoints = np.concatenate((q_waypoints[:i+1], [q_rand], q_waypoints[i+1:]), axis=0)
                    is_collision_check = True
                    break

        # Check again
        if is_collision(robot, q_matrix, faces, vertices, face_normals, collisions, env=env, return_once_found=False):
            print('Collision detected!')
        else:
            print('No collision found')
        
        input('Enter to finish\n')  
        env.close()


    def slider_callback(self, value, index, robot, labels):
        """
        This function is a callback from the Swift Slider elements. It uses the value
        returned by Swift, and encodes the robot joint that the slider is representing
        to update the robot's pose.
        """
        # Convert degrees to radians (slider in degrees, but robot joint in radians)
        radians = value * pi/180

        # Update robot.q (robot joints)
        robot.q[index] = radians

        # Calculate new DOF
        xyz, rpy = self.calculate_dof(robot)
        # Update labels
        labels[0].desc, labels[1].desc, labels[2].desc = f"X: {xyz[0]}", f"Y: {xyz[1]}", f"Z: {xyz[2]}"
        labels[3].desc, labels[4].desc, labels[5].desc = f"Roll (φ): {rpy[0]}", f"Pitch (θ): {rpy[1]}", f"Yaw (ψ): {rpy[2]}"


    def calculate_dof(self, robot):
        """
        This function accepts a robot, and returns the end-effector XYZ, RPY based
        on its current joint configuration.
        """
        ee_tr = robot.fkine(robot.q).A
        xyz = np.round(ee_tr[0:3, 3], 3)
        rpy = np.round(tr2rpy(ee_tr, unit="deg"), 2)

        return xyz, rpy
        
    
# --------------------------------------------------------------------------------------------------------------------------- #
def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv

    # Get and test parametric coords (s and t)
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:  # intersect_p is outside Triangle
        return 0

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:  # intersect_p is outside Triangle
        return False

    return True  # intersect_p is in Triangle


def is_collision(robot, q_matrix, faces, vertex, face_normals, collisions=[], env=None, return_once_found=True):
    """
    This is based upon the output of questions 2.5 and 2.6
    Given a robot model (robot), and trajectory (i.e. joint state vector) (q_matrix)
    and triangle obstacles in the environment (faces,vertex,face_normals)
    """
    result = False
    for i, q in enumerate(q_matrix):
        # Get the transform of every joint (i.e. start and end of every link)
        tr = get_link_poses(robot,q)
        
        # Go through each link and also each triangle face
        for i in range(np.size(tr,2)-1):
            for j, face in enumerate(faces):
                vert_on_plane = vertex[face][0]
                intersect_p, check = line_plane_intersection(face_normals[j], 
                                                            vert_on_plane, 
                                                            tr[i][:3,3], 
                                                            tr[i+1][:3,3])
                # list of all triangle combination in a face
                triangle_list  = np.array(list(combinations(face,3)),dtype= int)
                if check == 1:
                    for triangle in triangle_list:
                        if is_intersection_point_inside_triangle(intersect_p, vertex[triangle]):
                            # Create a red sphere in Swift at the intersection point IF environment passed - if lagging, reduce radius
                            if env is not None:
                                new_collision = Sphere(radius=0.05, color=[1.0, 0.0, 0.0, 1.0])
                                new_collision.T = transl(intersect_p[0], intersect_p[1], intersect_p[2])
                                env.add(new_collision)
                                collisions.append(new_collision)
                            result = True
                            if return_once_found:
                                return result
                            break
    return result


def get_link_poses(robot:DHRobot,q=None)->List[np.ndarray]|np.ndarray:
    """
    :param q robot joint angles
    :param robot -  seriallink robot model
    :param transforms - list of transforms
    """
    if q is None:
        return robot.fkine_all().A
    return robot.fkine_all(q).A


def fine_interpolation(q1, q2, max_step_radians = np.deg2rad(1))->np.ndarray:
    """
    Use results from Q2.6 to keep calling jtraj until all step sizes are
    smaller than a given max steps size
    """
    steps = 2
    while np.any(max_step_radians < np.abs(np.diff(jtraj(q1,q2,steps).q, axis= 0))):
        steps+=1
    return jtraj(q1,q2,steps).q


def interpolate_waypoints_radians(waypoint_radians, max_step_radians = np.deg2rad(1))->np.ndarray:
    """
    Given a set of waypoints, finely intepolate them
    """
    q_matrix = []
    for i in range(np.size(waypoint_radians,0)-1):
        for q in fine_interpolation(waypoint_radians[i], waypoint_radians[i+1], max_step_radians):
            q_matrix.append(q)
    return q_matrix


# --------------------------------------------------------------------------------------------------------------------------- #
# Main block
if __name__ == "__main__":
    soln = Lab5Solution()
    soln.question2_and_3()

    print("Lab 5, Questions 2 + 3 Solution completed. You can now close the window (if it didn't auto close).")