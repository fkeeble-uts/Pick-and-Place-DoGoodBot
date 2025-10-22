import numpy as np
import roboticstoolbox as rtb
import logging
from math import pi
from spatialmath import SE3

# Configure logging
logging.basicConfig(level=logging.INFO)


class RobotController:
    def __init__(self, env, scene, system_state):
        """
        Initialize robot controller.
        
        Args:
            env: Swift visualization environment
            scene: Scene object containing robot workspace
            system_state: SystemState object for E-STOP monitoring
        """
        self.env = env
        self.scene = scene
        self.system_state = system_state
        
        # Object manipulation state (Reverted to the original working variables)
        self.carried_object = None
        self.grasp_transform = None
        self.carried_object_name = None
        self.attachments = {}      # Parent -> [(child, transform)]
        # Keeping object_owner for consistency with SystemState file, but not using it 
        # for core pick/drop logic, which is managed by carried_object.
        self.object_owner = {}     # Object -> Robot mapping
    
    # ========================================================================
    # E-STOP & SAFETY
    # ========================================================================
    
    def _check_estop(self, robot) -> bool:
        """
        Check if E-STOP is active. Called in motion loops.
        
        Args:
            robot: Robot being controlled
            
        Returns:
            True if E-STOP active (motion should halt), False otherwise
        """
        if self.system_state.is_estop_active():
            print(f"\n E-STOP ACTIVE - [{robot.name}] Motion halted!")
            return True
        return False
    
    # ========================================================================
    # OBJECT MANIPULATION (Restored working logic)
    # ========================================================================
    
    def attach_objects(self, parent_obj, child_obj):
        if parent_obj is child_obj:
            print("Cannot attach an object to itself")
            return 
        # Find relative transform for object being attached
        T_world_parent = SE3(parent_obj.T)
        T_world_child = SE3(child_obj.T)
        T_rel = T_world_parent.inv() @ T_world_child
        # store the attachement 
        self.attachments.setdefault(parent_obj, []).append((child_obj, T_rel))
        print("Attatched {child_obj.name} to {parent_obj.name}")
        
    def pickup_object(self, robot, obj_to_pickup):
        if self.carried_object is None:
            T_world_tcp = SE3(robot.fkine(robot.q).A)
            T_world_object = SE3(obj_to_pickup.T)
            self.grasp_transform = T_world_tcp.inv() * T_world_object
            self.carried_object = obj_to_pickup
            object_type = type(obj_to_pickup).__name__
            self.carried_object_name = f"{object_type}"
            print(f"[{robot.name}] Picked up object ({self.carried_object_name}).")
        else:
            print("Already carrying an object.")

    def drop_object(self, robot):
        """'Detaches' the object from the robot and prints its final location."""
        if self.carried_object is not None:
            final_pose = SE3(self.carried_object.T)
            final_position = np.round(final_pose.t, 3)
            parent_name = self.carried_object_name     
            self.carried_object = None
            self.grasp_transform = None
            self.carried_object_name = None

            has_attachments = self.carried_object in self.attachments
            children_list = " (and attached objects)" if has_attachments else ""
            print(f"[{robot.name}] Dropped '{parent_name}'{children_list} at position: {final_position}")
        else:
             print(f"[{robot.name}] Not carrying any object to drop.")
    
    def _update_carried_object_pose(self, robot):
        """
        Update carried object position based on robot TCP. (Restored working logic)
        
        Args:
            robot: Robot holding object
        """
        if self.carried_object is not None:
            # Check if this robot is currently responsible for moving the carried object
            # In the original working logic, this check was implicit.
            # Here, we ensure the object is correctly registered to *this* robot if object_owner is used,
            # but we allow the update if only carried_object is set (like in the original working script)
            if self.object_owner.get(self.carried_object) == robot: 
                T_world_tcp = robot.fkine(robot.q)
                T_world_new_parent = T_world_tcp @ self.grasp_transform
                self.carried_object.T = T_world_new_parent.A
                
                # Update attached children
                if self.carried_object in self.attachments:
                    for child_obj, T_rel in self.attachments[self.carried_object]:
                        T_world_child = T_world_new_parent @ T_rel
                        child_obj.T = T_world_child.A
    
    # ========================================================================
    # INVERSE KINEMATICS (Kept E-STOP file's cleaner implementation)
    # ========================================================================
    
    def find_ikine(self, robot, target_tr, initial_q_guess=None, 
                   ignore_var="", ignore_rotation=False, hover_max=None):
        """
        Find inverse kinematics solution with custom constraints.
        
        Args:
            robot: Robot to solve IK for
            target_tr: Target SE3 transform
            initial_q_guess: Initial joint guess (uses random if None)
            ignore_var: Axis to ignore ('x', 'y', or 'z')
            ignore_rotation: If True, ignore orientation
            hover_max: Maximum hover distance for ignored axis
            
        Returns:
            tuple: (joint_solution, success_flag)
        """
        try:
            num_attempts = 300
            min_limits = robot.qlim[0, :]
            max_limits = robot.qlim[1, :]
            
            # Setup mask
            mask = [1, 1, 1, 1, 1, 1]
            coord_map = {"x": 0, "y": 1, "z": 2}
            
            if ignore_var in coord_map:
                mask[coord_map[ignore_var]] = 0
            if ignore_rotation:
                mask[3:6] = [0, 0, 0]
            
            # Iterative IK search
            for i in range(num_attempts):
                # Use initial guess on first attempt
                if i == 0 and initial_q_guess is not None:
                    q_guess = np.array(initial_q_guess)
                else:
                    q_guess = np.random.uniform(min_limits, max_limits)
                
                # Solve IK
                ik_result = robot.ikine_LM(target_tr, q0=q_guess, mask=mask)
                
                if ik_result.success:
                    solution = ik_result.q
                    
                    # Check joint limits
                    if np.all((solution >= min_limits) & (solution <= max_limits)):
                        
                        # Check hover constraint if specified
                        if hover_max is not None and ignore_var in coord_map:
                            axis_idx = coord_map[ignore_var]
                            sol_pose = robot.fkine(solution)
                            sol_coord = sol_pose.t[axis_idx]
                            target_coord = target_tr.t[axis_idx]
                            
                            lower = min(target_coord, target_coord + hover_max)
                            upper = max(target_coord, target_coord + hover_max)
                            
                            if lower <= sol_coord <= upper:
                                print(f"IK solution found on attempt {i+1} with valid hover on axis '{ignore_var}'.")
                                return solution, True
                        else:
                            return solution, True
            
            # Failed to find solution
            logging.warning(f"Failed to find a valid IK solution after {num_attempts} attempts.")
            return robot.q, False
        
        except Exception as e:
            logging.error(f"IK exception: {e}")
            return robot.q, False
    
    # ========================================================================
    # MOTION PRIMITIVES (Restored working logic and added E-STOP)
    # ========================================================================

    def move_rmrc(self, robot, target_pose, num_steps, delta_t=None, epsilon=0.1, lambda_max=0.1):
        """
        Moves the robot to a target pose using Resolved Motion Rate Control (RMRC).
        E-STOP functionality integrated.
        """
        if delta_t is None:
            delta_t = self.scene.SIM_STEP_TIME

        print(f"[{robot.name}] Following RMRC path...")

        # 1. Generate the Cartesian path using ctraj
        start_pose = robot.fkine(robot.q)
        cartesian_path = rtb.ctraj(start_pose, target_pose, num_steps)

        current_q = robot.q.copy()
        qlim = robot.qlim 

        # 2. RMRC Loop
        for i in range(num_steps - 1):
            
            # E-STOP check
            if self._check_estop(robot):
                return False, robot.q # Return failure and current configuration

            T_current = robot.fkine(current_q)
            T_desired = cartesian_path[i+1] # Target is the *next* pose in the path

            # Calculate velocity required to reach the next pose in delta_t
            delta_x = T_desired.t - T_current.t
            linear_velocity = delta_x / delta_t

            # Calculate angular velocity using the skew-symmetric matrix method
            Rd = T_desired.R
            Ra = T_current.R
            Rdot = (1/delta_t)*(Rd - Ra)
            S = Rdot @ Ra.T
            angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])

            # Desired end-effector velocity vector
            x_dot = np.concatenate((linear_velocity, angular_velocity))

            # Get Jacobian
            J = robot.jacob0(current_q)

            # Calculate manipulability and DLS damping factor
            try:
                m = np.sqrt(np.linalg.det(J @ J.T))
                if m < epsilon:
                    m_lambda = (1 - (m / epsilon)) * lambda_max
                else:
                    m_lambda = 0
            except np.linalg.LinAlgError: 
                print(f"Warning: Singularity detected (Jacobian determinant close to zero) at step {i+1}. Applying max damping.")
                m = 0
                m_lambda = lambda_max

            # Calculate Damped Least Squares Jacobian Inverse
            try:
                # DLS formula: J^T * inv(J * J^T + lambda * I)
                identity_matrix = np.eye(J.shape[0]) # 6x6 Identity
                inv_term = np.linalg.inv(J @ J.T + m_lambda * identity_matrix)
                inv_j = J.T @ inv_term
            except np.linalg.LinAlgError:
                print(f"Error: Could not compute inverse Jacobian at step {i+1}. Halting RMRC.")
                logging.error(f"Failed to compute inverse Jacobian at RMRC step {i+1}.")
                return False, robot.q # Return failure and current configuration

            # Calculate joint velocities
            q_dot = inv_j @ x_dot

            # Joint Limit Check
            q_next = current_q + q_dot * delta_t
            for j in range(robot.n):
                if q_next[j] < qlim[0, j]:
                    q_next[j] = qlim[0, j]
                    q_dot[j] = (q_next[j] - current_q[j]) / delta_t 
                elif q_next[j] > qlim[1, j]:
                    q_next[j] = qlim[1, j]
                    q_dot[j] = (q_next[j] - current_q[j]) / delta_t

            # Update state
            current_q = current_q + q_dot * delta_t
            robot.q = current_q
            self._update_carried_object_pose(robot)
            self.env.step(delta_t)

        print(f"✓ RMRC path complete.")
        return True, robot.q

    def wrap_to_near(self, q_goal, q_ref):
        """Wrap joint angles to be near reference configuration"""
        return q_ref + (q_goal - q_ref + pi) % (2 * pi) - pi

    def move_to_q(self, robot, q_target, steps=None, name=""):
        """
        Move to target joint configuration using joint trajectory (jtraj).
        E-STOP functionality integrated.
        
        Returns:
            tuple: (success, final_q)
        """
        if steps is None: steps = self.scene.TRAJ_STEPS
        
        q_start = robot.q.copy()
        # Use the original working utility function
        q_target = self.wrap_to_near(q_target, q_start)
        
        print(f"[{robot.name}] Moving to {name}...")
        trajectory = rtb.jtraj(q_start, q_target, steps)
        
        for q in trajectory.q:
            # E-STOP check
            if self._check_estop(robot):
                return False, robot.q
            
            robot.q = q
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)
            
        print(f"✓ {name}")
        return True, q_target

    def animate_trajectory(self, robot, start_q, end_q, steps):
        """
        Animate smooth joint-space trajectory with E-STOP checking.
        
        Returns:
            tuple: (success, final_q)
        """
        q_path = rtb.jtraj(start_q, end_q, steps).q
        
        for q_config in q_path:
            # E-STOP check
            if self._check_estop(robot):
                return False, robot.q
            
            # Apply joint limits
            robot.q = np.clip(q_config, robot.qlim[0, :], robot.qlim[1, :])
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)
        
        print("✓ Trajectory complete")
        return True, robot.q

    def print_pose(self, robot, label=""):
        """Print current robot pose for debugging"""
        print(f"\n{label}")
        q_deg = np.round(np.rad2deg(robot.q), 2).tolist()
        T = robot.fkine(robot.q)
        print(f"  Joints (deg): {q_deg}")
        print(f"  TCP Position: {np.round(T.t, 3)}")







import numpy as np
import roboticstoolbox as rtb
import logging
from math import pi
from spatialmath import SE3

class RobotController:
    def __init__(self, env, scene, system_state):
        self.env = env
        self.scene = scene
        self.system_state = system_state

        self.carried_object = None
        self.grasp_transform = None
        self.carried_object_name = None
        self.attachments = {}
        self.object_owner = {}   


    # ========================================================================
    # E-STOP & SAFETY
    # ========================================================================
    
    def _check_estop(self, robot) -> bool:
        if self.system_state.is_estop_active():
            print(f"\n E-STOP ACTIVE - [{robot.name}] Motion halted!")
            return True
        return False
    
    def attach_objects(self, parent_obj, child_obj):
        if parent_obj is child_obj:
            print("Cannot attach an object to itself")
            return 
        # Find relative transform for object being attached
        T_world_parent = SE3(parent_obj.T)
        T_world_child = SE3(child_obj.T)
        T_rel = T_world_parent.inv() @ T_world_child
        # store the attachement 
        self.attachments.setdefault(parent_obj, []).append((child_obj, T_rel))
        print("Attatched {child_obj.name} to {parent_obj.name}")

    def pickup_object(self, robot, obj_to_pickup):
        if self.carried_object is None:
            T_world_tcp = SE3(robot.fkine(robot.q).A)
            T_world_object = SE3(obj_to_pickup.T)
            self.grasp_transform = T_world_tcp.inv() * T_world_object
            self.carried_object = obj_to_pickup
            object_type = type(obj_to_pickup).__name__
            self.carried_object_name = f"{object_type}"
            print(f"[{robot.name}] Picked up object ({self.carried_object_name}).")
        else:
            print("Already carrying an object.")

    def drop_object(self, robot):
        """'Detaches' the object from the robot and prints its final location."""
        if self.carried_object is not None:
            final_pose = SE3(self.carried_object.T)
            final_position = np.round(final_pose.t, 3)
            parent_name = self.carried_object_name     
            self.carried_object = None
            self.grasp_transform = None
            self.carried_object_name = None

            has_attachments = self.carried_object in self.attachments
            children_list = " (and attached objects)" if has_attachments else ""
            print(f"[{robot.name}] Dropped '{parent_name}'{children_list} at position: {final_position}")
        else:
             print(f"[{robot.name}] Not carrying any object to drop.")

    def _update_carried_object_pose(self, robot):
        if self.carried_object is not None:
            T_world_tcp = robot.fkine(robot.q)
            T_world_new_parent = T_world_tcp @ self.grasp_transform
            self.carried_object.T = T_world_new_parent.A
            
            if self.carried_object in self.attachments:
                for child_obj, T_rel in self.attachments[self.carried_object]:
                    T_world_new_child = T_world_new_parent @ T_rel
                    child_obj.T = T_world_new_child.A


    def find_ikine(self, robot, target_tr, initial_q_guess=None, ignore_var="", ignore_rotation=False, hover_max=None):
        num_attempts = 300
        min_limits = robot.qlim[0, :]
        max_limits = robot.qlim[1, :]
        mask = [1, 1, 1, 1, 1, 1]
        coord_map = {"x": 0, "y": 1, "z": 2}
        if ignore_var in coord_map: mask[coord_map[ignore_var]] = 0
        if ignore_rotation: mask[3:6] = [0, 0, 0]
        for i in range(num_attempts):
            if i == 0 and initial_q_guess is not None: q_guess = initial_q_guess
            else: q_guess = np.random.uniform(low=min_limits, high=max_limits)
            ik_result = robot.ikine_LM(target_tr, q0=q_guess, mask=mask)
            if ik_result.success:
                solution = ik_result.q
                if np.all((solution >= min_limits) & (solution <= max_limits)):
                    if hover_max is not None and ignore_var in coord_map:
                        axis_index = coord_map[ignore_var]
                        solution_pose = robot.fkine(solution)
                        solution_coord = solution_pose.t[axis_index]
                        target_coord = target_tr.t[axis_index]
                        lower_bound = min(target_coord, target_coord + hover_max)
                        upper_bound = max(target_coord, target_coord + hover_max)
                        if lower_bound <= solution_coord <= upper_bound:
                            print(f"IK solution found on attempt {i+1} with valid hover on axis '{ignore_var}'.")
                            return solution, True
                    else:
                        return solution, True
        logging.warning(f"Failed to find a valid IK solution after {num_attempts} attempts.")
        return robot.q, False

    def move_rmrc(self, robot, target_pose, num_steps, delta_t=None, epsilon=0.1, lambda_max=0.1):
        # Moves the robot to a target pose using Resolved Motion Rate Control

        if delta_t is None:
            delta_t = self.scene.SIM_STEP_TIME

        print(f"[{robot.name}] Following RMRC path...")

        # 1. Generate the Cartesian path using ctraj
        start_pose = robot.fkine(robot.q)
        cartesian_path = rtb.ctraj(start_pose, target_pose, num_steps)

        current_q = robot.q.copy()
        qlim = robot.qlim # Get joint limits, transpose might not be needed depending on shape

        # 2. RMRC Loop
        for i in range(num_steps - 1):

            # E-STOP check
            if self._check_estop(robot):
                return False, robot.q
            
            T_current = robot.fkine(current_q)
            T_desired = cartesian_path[i+1] # Target is the *next* pose in the path

            # Calculate velocity required to reach the next pose in delta_t
            delta_x = T_desired.t - T_current.t
            linear_velocity = delta_x / delta_t

            # Calculate angular velocity using the skew-symmetric matrix method
            Rd = T_desired.R
            Ra = T_current.R
            Rdot = (1/delta_t)*(Rd - Ra)
            S = Rdot @ Ra.T
            # Ensure S is skew-symmetric (optional check, helps debugging)
            # S = (S - S.T) / 2
            angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])

            # Desired end-effector velocity vector
            x_dot = np.concatenate((linear_velocity, angular_velocity))

            # Get Jacobian
            J = robot.jacob0(current_q)

            # Calculate manipulability and DLS damping factor
            try:
                m = np.sqrt(np.linalg.det(J @ J.T))
                if m < epsilon:
                    m_lambda = (1 - (m / epsilon)) * lambda_max
                else:
                    m_lambda = 0
            except np.linalg.LinAlgError: # Handle cases where J@J.T might be singular
                 print(f"Warning: Singularity detected (Jacobian determinant close to zero) at step {i+1}. Applying max damping.")
                 m = 0
                 m_lambda = lambda_max


            # Calculate Damped Least Squares Jacobian Inverse
            try:
                # DLS formula: J^T * inv(J * J^T + lambda * I)
                identity_matrix = np.eye(J.shape[0]) # 6x6 Identity
                inv_term = np.linalg.inv(J @ J.T + m_lambda * identity_matrix)
                inv_j = J.T @ inv_term
            except np.linalg.LinAlgError:
                print(f"Error: Could not compute inverse Jacobian at step {i+1}. Halting RMRC.")
                logging.error(f"Failed to compute inverse Jacobian at RMRC step {i+1}.")
                return (False, robot.q) # Return current state

            # Calculate joint velocities
            q_dot = inv_j @ x_dot

            # Joint Limit Check
            q_next = current_q + q_dot * delta_t
            for j in range(robot.n):
                if q_next[j] < qlim[0, j]:
                    # Clip to limit and effectively stop motion for this joint
                    q_next[j] = qlim[0, j]
                    q_dot[j] = (q_next[j] - current_q[j]) / delta_t # Adjust q_dot based on clipping
                    # print(f"Warning: Joint {j+1} hit lower limit at step {i+1}.")
                elif q_next[j] > qlim[1, j]:
                    q_next[j] = qlim[1, j]
                    q_dot[j] = (q_next[j] - current_q[j]) / delta_t
                    # print(f"Warning: Joint {j+1} hit upper limit at step {i+1}.")

            # Update state
            current_q = current_q + q_dot * delta_t # Recalculate q_next after potential clipping adjustment
            robot.q = current_q
            self._update_carried_object_pose(robot)
            self.env.step(delta_t)

        print(f"✓ RMRC path complete.")
        return True, robot.q

    def wrap_to_near(self, q_goal, q_ref):
        return q_ref + (q_goal - q_ref + pi) % (2 * pi) - pi

    def move_to_q(self, robot, q_target, steps=None, name=""):
        if steps is None: steps = self.scene.TRAJ_STEPS
        q_start = robot.q.copy()
        q_target = self.wrap_to_near(q_target, q_start)
        print(f"[{robot.name}] Moving to {name}...")
        trajectory = rtb.jtraj(q_start, q_target, steps)
        for q in trajectory.q:
            if self._check_estop(robot):
                return False, robot.q
            robot.q = q
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)
        print(f"✓ {name}")
        return True, q_target

    def animate_trajectory(self, robot, start_q, end_q, steps):
        q_path = rtb.jtraj(start_q, end_q, steps).q
        for q_config in q_path:
            if self._check_estop(robot):
                return False, robot.q
            robot.q = np.clip(q_config, robot.qlim[0, :], robot.qlim[1, :])
            self._update_carried_object_pose(robot)
            self.env.step(self.scene.SIM_STEP_TIME)
        print("✓ Trajectory complete")
        return True, robot.q

    def print_pose(self, robot, label=""):
        print(f"\n{label}")
        q_deg = np.round(np.rad2deg(robot.q), 2)
        q_deg = q_deg.tolist()
        T = robot.fkine(robot.q)
        print(f"  Joints (deg): {q_deg}")
        print(f"  TCP Pos: {np.round(T.t, 3)}")