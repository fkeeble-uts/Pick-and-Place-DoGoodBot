"""
gui.py
GUI interface for robot bartender system.
Provides manual control, sequence execution, and E-STOP functionality.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from SystemState import SystemState, RobotState, SequenceProgress
from main import setup_robot_system, run_sequence_worker
import keyboard


class RobotBartenderGUI:
    """Main GUI application for robot bartender control"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Bartender Control System")
        self.root.geometry("1400x900")
        
        # System components
        self.system_state = SystemState()
        self.progress = SequenceProgress()
        self.env = None
        self.scene = None
        self.controller = None
        self.robots = None
        self.worker_thread = None
        
        # GUI state
        self.update_interval = 100  # ms
        self.control_mode = "joint"  # "joint" or "cartesian"
        self.jog_mode = "step"  # "step" or "slider"
        
        # Build GUI
        self._create_widgets()
        self._setup_layout()
        
        # Initialize system in background
        self.root.after(100, self._initialize_system)
    
    # ========================================================================
    # GUI CONSTRUCTION
    # ========================================================================
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # ===== TOP FRAME: System Status =====
        self.status_frame = ttk.LabelFrame(self.root, text="System Status", padding=10)
        
        self.state_label = ttk.Label(
            self.status_frame,
            text="State: INITIALIZING",
            font=("Arial", 16, "bold")
        )
        
        self.progress_label = ttk.Label(
            self.status_frame,
            text="Progress: Not started",
            font=("Arial", 12)
        )
        
        # ===== LEFT FRAME: Robot Control =====
        self.control_frame = ttk.LabelFrame(self.root, text="Robot Control", padding=10)
        
        # Main control buttons
        self.start_button = ttk.Button(
            self.control_frame,
            text="START SEQUENCE",
            command=self._on_start,
            state=tk.DISABLED
        )
        
        self.estop_button = tk.Button(
            self.control_frame,
            text="EMERGENCY STOP",
            command=self._on_estop,
            bg="red",
            fg="white",
            font=("Arial", 14, "bold"),
            height=2,
            state=tk.DISABLED
        )
        
        self.disarm_button = ttk.Button(
            self.control_frame,
            text="DISARM E-STOP",
            command=self._on_disarm,
            state=tk.DISABLED
        )
        
        self.resume_button = ttk.Button(
            self.control_frame,
            text="RESUME SEQUENCE",
            command=self._on_resume,
            state=tk.DISABLED
        )
        
        self.clear_button = ttk.Button(
            self.control_frame,
            text="Clear Progress",
            command=self._on_clear_progress,
            state=tk.DISABLED
        )
        
        # Control mode selection
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        ttk.Label(self.control_frame, text="Control Mode:", font=("Arial", 11, "bold")).pack(pady=5)
        
        self.control_mode_var = tk.StringVar(value="joint")
        
        ttk.Radiobutton(
            self.control_frame,
            text="Joint Control",
            variable=self.control_mode_var,
            value="joint",
            command=self._on_control_mode_change
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        ttk.Radiobutton(
            self.control_frame,
            text="Cartesian Control",
            variable=self.control_mode_var,
            value="cartesian",
            command=self._on_control_mode_change
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # ===== CENTER FRAME: Manual Control =====
        self.manual_frame = ttk.LabelFrame(self.root, text="Manual Control (Teach Mode)", padding=10)
        
        # Robot selection (always visible)
        ttk.Label(self.manual_frame, text="Select Robot:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.robot_var = tk.StringVar()
        self.robot_combo = ttk.Combobox(
            self.manual_frame,
            textvariable=self.robot_var,
            state="readonly",
            width=25
        )
        self.robot_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=(10, 0))
        
        # Jog mode selection (Step vs Slider)
        ttk.Label(self.manual_frame, text="Jog Mode:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.jog_mode_var = tk.StringVar(value="step")
        mode_frame = ttk.Frame(self.manual_frame)
        mode_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5, padx=(10, 0))
        
        ttk.Radiobutton(
            mode_frame, text="Step", variable=self.jog_mode_var, 
            value="step", command=self._on_jog_mode_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Slider", variable=self.jog_mode_var,
            value="slider", command=self._on_jog_mode_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(self.manual_frame, orient=tk.HORIZONTAL).grid(
            row=2, column=0, columnspan=3, sticky=tk.EW, pady=10
        )
        
        # ===== JOINT CONTROL PANEL =====
        self.joint_control_frame = ttk.Frame(self.manual_frame)
        self.joint_control_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW)
        
        # Create 6 joint controls
        self.joint_sliders = []
        self.joint_step_buttons = []
        self.joint_value_labels = []
        
        for i in range(6):
            # Joint label
            ttk.Label(
                self.joint_control_frame, 
                text=f"Joint {i+1}:",
                font=("Arial", 10, "bold")
            ).grid(row=i*2, column=0, sticky=tk.W, pady=(10, 2))
            
            # Value display
            value_label = ttk.Label(self.joint_control_frame, text="0.0°")
            value_label.grid(row=i*2, column=1, sticky=tk.W, padx=(10, 0))
            self.joint_value_labels.append(value_label)
            
            # Slider
            slider = tk.Scale(
                self.joint_control_frame,
                from_=-180,
                to=180,
                orient=tk.HORIZONTAL,
                length=400,
                resolution=0.5,
                showvalue=False,
                command=lambda val, idx=i: self._on_joint_slider_live_move(idx, val),                
                state=tk.DISABLED
            )
            slider.grid(row=i*2+1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
            self.joint_sliders.append(slider)
            
            # Step buttons
            step_frame = ttk.Frame(self.joint_control_frame)
            step_frame.grid(row=i*2+1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
            
            ttk.Label(step_frame, text="Step (deg):").pack(side=tk.LEFT, padx=(0, 5))
            
            step_entry = ttk.Entry(step_frame, width=8)
            step_entry.insert(0, "5")
            step_entry.pack(side=tk.LEFT, padx=5)
            
            btn_neg = ttk.Button(
                step_frame, text="<", width=3,
                command=lambda idx=i, entry=step_entry: self._on_joint_step(idx, entry, -1),
                state=tk.DISABLED
            )
            btn_neg.pack(side=tk.LEFT, padx=2)
            
            btn_pos = ttk.Button(
                step_frame, text=">", width=3,
                command=lambda idx=i, entry=step_entry: self._on_joint_step(idx, entry, 1),
                state=tk.DISABLED
            )
            btn_pos.pack(side=tk.LEFT, padx=2)
            
            self.joint_step_buttons.append((btn_neg, btn_pos, step_entry, step_frame))
        
        # Hide step buttons initially (will be shown/hidden based on mode)
        self._update_jog_mode_display()
        
        # ===== CARTESIAN CONTROL PANEL =====
        self.cartesian_control_frame = ttk.Frame(self.manual_frame)
        # Don't grid initially - will be shown when cartesian mode selected
        
        # Note: Using base frame only
        self.frame_var = tk.StringVar(value="base")
        
        ttk.Separator(self.cartesian_control_frame, orient=tk.HORIZONTAL).grid(
            row=0, column=0, columnspan=3, sticky=tk.EW, pady=10
        )
        
        # Create controls for X, Y, Z
        self.cart_sliders = []
        self.cart_step_buttons = []
        self.cart_value_labels = []
        
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            # Axis label
            ttk.Label(
                self.cartesian_control_frame, 
                text=f"{axis} Position:",
                font=("Arial", 10, "bold")
            ).grid(row=i*2+1, column=0, sticky=tk.W, pady=(10, 2))
            
            # Value display
            value_label = ttk.Label(self.cartesian_control_frame, text="0.000 m")
            value_label.grid(row=i*2+1, column=1, sticky=tk.W, padx=(10, 0))
            self.cart_value_labels.append(value_label)
            
            # Slider (for future use if needed)
            slider = tk.Scale(
                self.cartesian_control_frame,
                from_=-1.0,
                to=1.0,
                orient=tk.HORIZONTAL,
                length=400,
                resolution=0.001,
                showvalue=False,
                state=tk.DISABLED
            )
            # Not adding to grid - sliders disabled for cartesian for now
            self.cart_sliders.append(slider)
            
            # Step buttons
            step_frame = ttk.Frame(self.cartesian_control_frame)
            step_frame.grid(row=i*2+2, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
            
            ttk.Label(step_frame, text="Step (mm):").pack(side=tk.LEFT, padx=(0, 5))
            
            step_entry = ttk.Entry(step_frame, width=8)
            step_entry.insert(0, "5")
            step_entry.pack(side=tk.LEFT, padx=5)
            
            btn_neg = ttk.Button(
                step_frame, text="<", width=3,
                command=lambda ax=axis.lower(): self._on_cart_step(ax + '-', None),
                state=tk.DISABLED
            )
            btn_neg.pack(side=tk.LEFT, padx=2)
            
            btn_pos = ttk.Button(
                step_frame, text=">", width=3,
                command=lambda ax=axis.lower(): self._on_cart_step(ax + '+', None),
                state=tk.DISABLED
            )
            btn_pos.pack(side=tk.LEFT, padx=2)
            
            self.cart_step_buttons.append((btn_neg, btn_pos, step_entry, step_frame))
        
        # TCP Position display
        ttk.Separator(self.cartesian_control_frame, orient=tk.HORIZONTAL).grid(
            row=10, column=0, columnspan=3, sticky=tk.EW, pady=10
        )
        ttk.Label(self.cartesian_control_frame, text="Current TCP Position:", font=("Arial", 10, "bold")).grid(
            row=11, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        self.tcp_pos_label = ttk.Label(self.cartesian_control_frame, text="X: 0.000  Y: 0.000  Z: 0.000")
        self.tcp_pos_label.grid(row=12, column=0, columnspan=3, sticky=tk.W, padx=10)
        
        # ===== RIGHT FRAME: Log Output =====
        self.log_frame = ttk.LabelFrame(self.root, text="System Log", padding=10)
        
        # Log text area with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            self.log_frame,
            height=30,
            width=45,
            state=tk.DISABLED,
            wrap=tk.WORD,
            yscrollcommand=log_scroll.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Clear log button
        self.clear_log_button = ttk.Button(
            self.log_frame,
            text="Clear Log",
            command=self._clear_log
        )
    
    def _setup_layout(self):
        """Arrange widgets in the window"""
        
        # Top status bar
        self.status_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        self.state_label.pack(pady=5)
        self.progress_label.pack(pady=5)
        
        # Left panel - Control buttons
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.start_button.pack(pady=10, fill=tk.X)
        self.estop_button.pack(pady=10, fill=tk.X)
        self.disarm_button.pack(pady=5, fill=tk.X)
        self.resume_button.pack(pady=5, fill=tk.X)
        self.clear_button.pack(pady=5, fill=tk.X)
        
        # Center panel - Manual controls
        self.manual_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right panel - Log output
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        self.clear_log_button.pack(pady=5)
    
    # ========================================================================
    # MODE SWITCHING
    # ========================================================================
    
    def _on_control_mode_change(self):
        """Handle switching between joint and cartesian control"""
        mode = self.control_mode_var.get()
        
        if mode == "joint":
            self.joint_control_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW)
            self.cartesian_control_frame.grid_remove()
        else:  # cartesian
            self.joint_control_frame.grid_remove()
            self.cartesian_control_frame.grid(row=3, column=0, columnspan=3, sticky=tk.NSEW)
        
        self._update_jog_mode_display()
    
    def _on_jog_mode_change(self):
        """Handle switching between step and slider jog modes"""
        self._update_jog_mode_display()
        
        # Update sliders to match current positions
        if self.jog_mode_var.get() == "slider":
            self._update_sliders_from_robot()
    
    def _update_jog_mode_display(self):
        """Show/hide step buttons and sliders based on jog mode"""
        mode = self.jog_mode_var.get()
        
        if self.control_mode_var.get() == "joint":
            # Joint control
            for i, (slider, (btn_neg, btn_pos, entry, frame)) in enumerate(zip(self.joint_sliders, self.joint_step_buttons)):
                if mode == "step":
                    slider.grid_remove()
                    frame.grid(row=i*2+1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
                else:  # slider
                    frame.grid_remove()
                    slider.grid(row=i*2+1, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
        else:
            # Cartesian control - only step mode for now
            for btn_neg, btn_pos, entry, frame in self.cart_step_buttons:
                frame.grid()
    
    def _update_sliders_from_robot(self):
        """Update all sliders to match current robot joint angles"""
        if not self.robots or not self.robot_var.get():
            return
        
        try:
            robot = self.robots[self.robot_var.get()]
            for i, slider in enumerate(self.joint_sliders):
                angle_deg = robot.q[i] * (180.0 / 3.14159)
                slider.set(angle_deg)
        except:
            pass

    def _on_joint_slider_live_move(self, joint_idx, value):
        """
        Handle joint slider value change, immediately moving the robot
        to the target angle using the controller's motion command.
        """
        if self.system_state.state != RobotState.TEACH:
            # The slider command fires on drag, so we check the state here
            return

        try:
            robot_name = self.robot_var.get()
            if not robot_name:
                # If no robot is selected, don't try to move
                return

            robot = self.robots[robot_name]
            # Convert the slider value (degrees) to radians
            target_angle_rad = float(value) * (3.14159 / 180.0)
            
            # Check limits 
            if target_angle_rad < robot.qlim[0, joint_idx] or target_angle_rad > robot.qlim[1, joint_idx]:
                return
            
            # Create the target joint vector (q)
            target_q = robot.q.copy()
            target_q[joint_idx] = target_angle_rad

            if self.controller:
                robot.q = target_q # Directly set the joint angle
                self.controller._update_carried_object_pose(robot) # Update visualization
                self.env.step(0.02) # Step the simulator
                
        except Exception as e:
            # print(f"Slider live move error: {e}") # Keep logging but don't stop the program
            pass # Keep it quiet during fast dragging
    
    def _on_joint_step(self, joint_idx, step_entry, direction):
        """Handle joint step button press"""
        if self.system_state.state != RobotState.TEACH:
            messagebox.showwarning("Jog Error", "Jogging only available in TEACH mode")
            return
        
        try:
            robot_name = self.robot_var.get()
            if not robot_name:
                messagebox.showwarning("No Robot", "Please select a robot")
                return
            
            robot = self.robots[robot_name]
            
            # Get step size
            step_deg = float(step_entry.get())
            step_rad = step_deg * direction * (3.14159 / 180.0)
            
            # Apply jog
            new_q = robot.q.copy()
            new_q[joint_idx] += step_rad
            
            # Check limits
            if new_q[joint_idx] < robot.qlim[0, joint_idx]:
                messagebox.showwarning("Limit", f"Joint {joint_idx+1} at lower limit")
                return
            if new_q[joint_idx] > robot.qlim[1, joint_idx]:
                messagebox.showwarning("Limit", f"Joint {joint_idx+1} at upper limit")
                return
            
            # Apply new position
            robot.q = new_q
            if self.controller:
                self.controller._update_carried_object_pose(robot)
            if self.env:
                self.env.step(0.02)
            
            self._log(f"Jogged {robot_name} J{joint_idx+1} by {step_deg*direction:.1f} deg")
            
        except ValueError:
            messagebox.showerror("Input Error", "Invalid step size")
        except Exception as e:
            messagebox.showerror("Jog Error", f"Error during jog: {e}")
    
    def _on_cart_step(self, direction, step_entry_override):
        """Handle cartesian step button press"""
        if self.system_state.state != RobotState.TEACH:
            messagebox.showwarning("Jog Error", "Jogging only available in TEACH mode")
            return
        
        try:
            robot_name = self.robot_var.get()
            if not robot_name:
                messagebox.showwarning("No Robot", "Please select a robot")
                return
            
            robot = self.robots[robot_name]
            
            # Get step size from the appropriate entry
            axis_idx = 0 if 'x' in direction else (1 if 'y' in direction else 2)
            _, _, step_entry, _ = self.cart_step_buttons[axis_idx]
            step_mm = float(step_entry.get())
            step_m = step_mm / 1000.0
            
            # Get frame
            frame = self.frame_var.get()
            
            # Call controller's cartesian jog
            success, final_q = self.controller.jog_cartesian(
                robot, direction, step_size=step_m, num_steps=20, frame=frame
            )
            
            if success:
                self._log(f"Jogged {robot_name} {direction} by {step_mm:.1f}mm ({frame} frame)")
            else:
                self._log(f"Cartesian jog {direction} failed")
                messagebox.showwarning("Jog Failed", "Cannot reach target position.\nTry smaller step or different configuration.")
                
        except ValueError:
            messagebox.showerror("Input Error", "Invalid step size")
        except Exception as e:
            messagebox.showerror("Jog Error", f"Error during cartesian jog: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # SYSTEM INITIALIZATION
    # ========================================================================
    
    def _initialize_system(self):
        """Initialize robot system in background"""
        self._log("Initializing robot system...")
        
        def init_worker():
            try:
                self.env, self.scene, self.controller, self.robots = setup_robot_system(
                    self.system_state
                )
                
                # Update GUI on main thread
                self.root.after(0, self._on_initialization_complete)
                
            except Exception as ex:
                error_msg = str(ex)
                import traceback
                error_trace = traceback.format_exc()
                print(f"Initialization error:\n{error_trace}")
                self.root.after(0, lambda msg=error_msg: self._on_initialization_error(msg))
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def _on_initialization_complete(self):
        """Called when system initialization completes"""
        self._log("System initialization complete!")
        
        # Populate robot dropdown
        robot_names = list(self.robots.keys())
        self.robot_combo['values'] = robot_names
        if robot_names:
            self.robot_combo.current(0)
        
        # Enable controls
        self.start_button['state'] = tk.NORMAL
        self.estop_button['state'] = tk.NORMAL
        
        # Start GUI update loop
        self._update_gui()
        self._start_keyboard_listener()
    
    def _on_initialization_error(self, error_msg):
        """Called if initialization fails"""
        self._log(f"Initialization failed: {error_msg}")
        messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{error_msg}")
    
    # ========================================================================
    # BUTTON CALLBACKS
    # ========================================================================
    
    def _on_start(self):
        """Start button callback"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=run_sequence_worker,
                args=(self.controller, self.robots, self.scene, self.system_state, self.progress),
                daemon=True
            )
            self.worker_thread.start()
            self._log("Starting sequence...")
        
        self.system_state.set_state(RobotState.RUNNING)
        self._log("System state: RUNNING")
    
    def _on_estop(self):
        """E-STOP button callback"""
        self.system_state.set_state(RobotState.ESTOP_ACTIVE)
        self._log("EMERGENCY STOP ACTIVATED!")
        messagebox.showwarning("E-STOP", "Emergency Stop Activated!\nAll motion halted.")
    
    def _on_disarm(self):
        """Disarm E-STOP button callback"""
        self.progress.reset()
        self.system_state.set_state(RobotState.TEACH)
        self._log("E-STOP disarmed - system reset")
    
    def _on_resume(self):
        """Resume button callback"""
        self.system_state.set_state(RobotState.RUNNING)
        self._log(f"Resuming from: {self.progress.get_status()}")
    
    def _on_clear_progress(self):
        """Clear progress button callback"""
        if messagebox.askyesno("Clear Progress", "Reset sequence progress to start?"):
            self.progress.reset()
            self._log("Progress cleared - will restart from beginning")
    
    # ========================================================================
    # GUI UPDATE LOOP
    # ========================================================================
    
    def _update_gui(self):
        """Periodic GUI update (called every update_interval ms)"""
        
        # Update state display
        state = self.system_state.state
        self.state_label['text'] = f"State: {state.name}"
        
        # Color code state
        if state == RobotState.RUNNING:
            self.state_label['foreground'] = 'green'
        elif state == RobotState.ESTOP_ACTIVE:
            self.state_label['foreground'] = 'red'
        elif state == RobotState.PAUSED:
            self.state_label['foreground'] = 'orange'
        else:
            self.state_label['foreground'] = 'black'
        
        # Update progress display
        self.progress_label['text'] = f"Progress: {self.progress.get_status()}"
        
        # Update button states based on system state
        teach_mode = (state == RobotState.TEACH)
        button_state = tk.NORMAL if teach_mode else tk.DISABLED
        
        # Enable/disable all joint control buttons
        for slider in self.joint_sliders:
            slider['state'] = button_state
        for btn_neg, btn_pos, entry, frame in self.joint_step_buttons:
            btn_neg['state'] = button_state
            btn_pos['state'] = button_state
        
        # Enable/disable all cartesian control buttons
        for btn_neg, btn_pos, entry, frame in self.cart_step_buttons:
            btn_neg['state'] = button_state
            btn_pos['state'] = button_state
        
        # Update main control buttons
        if state == RobotState.TEACH:
            self.start_button['state'] = tk.NORMAL
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.NORMAL
            
        elif state == RobotState.RUNNING:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED
            
        elif state == RobotState.ESTOP_ACTIVE:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.DISABLED
            self.disarm_button['state'] = tk.NORMAL
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED
            
        elif state == RobotState.PAUSED:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.NORMAL
            self.clear_button['state'] = tk.NORMAL
        
        # Update joint angles and TCP position display
        if self.robots and self.robot_var.get():
            robot = self.robots[self.robot_var.get()]
            
            # Update joint value labels
            for i, label in enumerate(self.joint_value_labels):
                angle_deg = robot.q[i] * (180.0 / 3.14159)
                label['text'] = f"{angle_deg:7.2f}°"
            
            # Update TCP position
            tcp_pose = robot.fkine(robot.q)
            for i, label in enumerate(self.cart_value_labels):
                label['text'] = f"{tcp_pose.t[i]:.4f} m"
            
            self.tcp_pos_label['text'] = f"X: {tcp_pose.t[0]:.3f}  Y: {tcp_pose.t[1]:.3f}  Z: {tcp_pose.t[2]:.3f}"
        
        # Update Swift environment
        if self.env:
            self.env.step(0.02)
        
        # Schedule next update
        self.root.after(self.update_interval, self._update_gui)

    # ========================================================================
    # KEYBOARD LISTENER
    # ========================================================================
    
    def _start_keyboard_listener(self):
        """Starts a thread to listen for the 'E' keypress for E-STOP."""
        
        def listener_worker():
            self._log("Terminal E-STOP listener active. Press 'e' to trigger.")
            while self.system_state.state != RobotState.QUIT:
                try:
                    keyboard.wait('e') 
                    
                    if self.system_state.state != RobotState.ESTOP_ACTIVE:
                        self.root.after(0, self._on_estop)
                    
                    time.sleep(0.5) 
                except Exception as e:
                    print(f"Keyboard listener error: {e}")
                    break

        threading.Thread(target=listener_worker, daemon=True).start()
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def _log(self, message):
        """Add message to log display"""
        self.log_text['state'] = tk.NORMAL
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text['state'] = tk.DISABLED
        print(message)
    
    def _clear_log(self):
        """Clear log display"""
        self.log_text['state'] = tk.NORMAL
        self.log_text.delete(1.0, tk.END)
        self.log_text['state'] = tk.DISABLED
    
    # ========================================================================
    # RUN
    # ========================================================================
    
    def run(self):
        """Start the GUI main loop"""
        self._log("GUI started - waiting for initialization...")
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _on_closing(self):
        """Handle window close event"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.system_state.set_state(RobotState.QUIT)
            self.root.destroy()


if __name__ == "__main__":
    gui = RobotBartenderGUI()
    gui.run()