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
        self.root.geometry("1200x800")
        
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
            text="🛑 EMERGENCY STOP 🛑",
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
        
        # ===== CENTER FRAME: Robot Selection & Jogging =====
        self.jog_frame = ttk.LabelFrame(self.root, text="Manual Control (Teach Mode)", padding=10)
        
        # Robot selection
        ttk.Label(self.jog_frame, text="Select Robot:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.robot_var = tk.StringVar()
        self.robot_combo = ttk.Combobox(
            self.jog_frame,
            textvariable=self.robot_var,
            state="readonly",
            width=25
        )
        self.robot_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5)
        
        # Joint selection
        ttk.Label(self.jog_frame, text="Select Joint:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.joint_var = tk.StringVar()
        self.joint_combo = ttk.Combobox(
            self.jog_frame,
            textvariable=self.joint_var,
            values=[f"Joint {i+1}" for i in range(6)],
            state="readonly",
            width=25
        )
        self.joint_combo.current(0)
        self.joint_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=5)
        
        # Jog step size
        ttk.Label(self.jog_frame, text="Step Size (deg):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.step_var = tk.StringVar(value="2")
        self.step_entry = ttk.Entry(self.jog_frame, textvariable=self.step_var, width=10)
        self.step_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Jog buttons
        jog_button_frame = ttk.Frame(self.jog_frame)
        jog_button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.jog_neg_button = ttk.Button(
            jog_button_frame,
            text="◀ Jog -",
            command=lambda: self._on_jog(-1),
            state=tk.DISABLED
        )
        self.jog_neg_button.pack(side=tk.LEFT, padx=5)
        
        self.jog_pos_button = ttk.Button(
            jog_button_frame,
            text="Jog + ▶",
            command=lambda: self._on_jog(1),
            state=tk.DISABLED
        )
        self.jog_pos_button.pack(side=tk.LEFT, padx=5)
        
        # Current joint positions display
        ttk.Separator(self.jog_frame, orient=tk.HORIZONTAL).grid(
            row=4, column=0, columnspan=3, sticky=tk.EW, pady=10
        )
        
        ttk.Label(self.jog_frame, text="Current Joint Positions:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        
        self.joint_labels = []
        for i in range(6):
            label = ttk.Label(self.jog_frame, text=f"J{i+1}: 0.0°")
            label.grid(row=6+i, column=0, columnspan=3, sticky=tk.W, padx=10)
            self.joint_labels.append(label)
        
        # ===== RIGHT FRAME: Log Output =====
        self.log_frame = ttk.LabelFrame(self.root, text="System Log", padding=10)
        
        # Log text area with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            self.log_frame,
            height=35,
            width=50,
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
        
        # Center panel - Jogging controls
        self.jog_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        # Right panel - Log output
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.clear_log_button.pack(pady=5)
    
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
                # Capture exception message before lambda
                error_msg = str(ex)
                import traceback
                error_trace = traceback.format_exc()
                print(f"Initialization error:\n{error_trace}")
                self.root.after(0, lambda msg=error_msg: self._on_initialization_error(msg))
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def _on_initialization_complete(self):
        """Called when system initialization completes"""
        self._log("✅ System initialization complete!")
        
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
        self._log(f"❌ Initialization failed: {error_msg}")
        messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{error_msg}")
    
    # ========================================================================
    # BUTTON CALLBACKS
    # ========================================================================
    
    def _on_start(self):
        """Start button callback"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            # Start new worker thread
            self.worker_thread = threading.Thread(
                target=run_sequence_worker,
                args=(self.controller, self.robots, self.scene, self.system_state, self.progress),
                daemon=True
            )
            self.worker_thread.start()
            self._log("🚀 Starting sequence...")
        
        # Set state to RUNNING
        self.system_state.set_state(RobotState.RUNNING)
        self._log("System state: RUNNING")
    
    def _on_estop(self):
        """E-STOP button callback"""
        self.system_state.set_state(RobotState.ESTOP_ACTIVE)
        self._log("🛑 EMERGENCY STOP ACTIVATED!")
        messagebox.showwarning("E-STOP", "Emergency Stop Activated!\nAll motion halted.")
    
    def _on_disarm(self):
        """Disarm E-STOP button callback"""
        self.progress.reset()
        self.system_state.set_state(RobotState.TEACH)
        self._log("E-STOP disarmed - system reset")
    
    def _on_resume(self):
        """Resume button callback"""
        self.system_state.set_state(RobotState.RUNNING)
        self._log(f"▶️  Resuming from: {self.progress.get_status()}")
    
    def _on_clear_progress(self):
        """Clear progress button callback"""
        if messagebox.askyesno("Clear Progress", "Reset sequence progress to start?"):
            self.progress.reset()
            self._log("🔄 Progress cleared - will restart from beginning")
    
    def _on_jog(self, direction):
        """Jog button callback"""
        if self.system_state.state != RobotState.TEACH:
            messagebox.showwarning("Jog Error", "Jogging only available in TEACH mode")
            return
        
        try:
            # Get selected robot
            robot_name = self.robot_var.get()
            if not robot_name:
                messagebox.showwarning("No Robot", "Please select a robot")
                return
            
            robot = self.robots[robot_name]
            
            # Get selected joint (0-indexed)
            joint_str = self.joint_var.get()
            joint_idx = int(joint_str.split()[-1]) - 1
            
            # Get step size
            step_deg = float(self.step_var.get())
            step_rad = step_deg * direction * (3.14159 / 180.0)
            
            # Apply jog
            new_q = robot.q.copy()
            new_q[joint_idx] += step_rad
            
            # Check limits
            if new_q[joint_idx] < robot.qlim[0, joint_idx]:
                messagebox.showwarning("Limit", "Joint at lower limit")
                return
            if new_q[joint_idx] > robot.qlim[1, joint_idx]:
                messagebox.showwarning("Limit", "Joint at upper limit")
                return
            
            # Apply new position
            robot.q = new_q
            self.env.step(0.02)
            
            self._log(f"Jogged {robot_name} J{joint_idx+1} by {step_deg*direction:.1f}°")
            
        except ValueError:
            messagebox.showerror("Input Error", "Invalid step size")
        except Exception as e:
            messagebox.showerror("Jog Error", f"Error during jog: {e}")
    
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
        if state == RobotState.TEACH:
            self.start_button['state'] = tk.NORMAL
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.NORMAL
            self.jog_neg_button['state'] = tk.NORMAL
            self.jog_pos_button['state'] = tk.NORMAL
            
        elif state == RobotState.RUNNING:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED
            self.jog_neg_button['state'] = tk.DISABLED
            self.jog_pos_button['state'] = tk.DISABLED
            
        elif state == RobotState.ESTOP_ACTIVE:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.DISABLED
            self.disarm_button['state'] = tk.NORMAL
            self.resume_button['state'] = tk.DISABLED
            self.clear_button['state'] = tk.DISABLED
            self.jog_neg_button['state'] = tk.DISABLED
            self.jog_pos_button['state'] = tk.DISABLED
            
        elif state == RobotState.PAUSED:
            self.start_button['state'] = tk.DISABLED
            self.estop_button['state'] = tk.NORMAL
            self.disarm_button['state'] = tk.DISABLED
            self.resume_button['state'] = tk.NORMAL
            self.clear_button['state'] = tk.NORMAL
            self.jog_neg_button['state'] = tk.DISABLED
            self.jog_pos_button['state'] = tk.DISABLED
        
        # Update joint position display
        if self.robots and self.robot_var.get():
            robot = self.robots[self.robot_var.get()]
            for i, label in enumerate(self.joint_labels):
                angle_deg = robot.q[i] * (180.0 / 3.14159)
                label['text'] = f"J{i+1}: {angle_deg:6.1f}°"
        
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
            # The keyboard.wait() function is blocking, so it must run in a thread.
            # It blocks the execution until the specified key is pressed.
            self._log("Terminal E-STOP listener active. Press 'e' to trigger.")
            while self.system_state.state != RobotState.QUIT:
                try:
                    # Wait for the 'e' key to be pressed
                    keyboard.wait('e') 
                    
                    # Ensure the E-STOP is not already active
                    if self.system_state.state != RobotState.ESTOP_ACTIVE:
                        # Use root.after to safely call the GUI method from the thread
                        self.root.after(0, self._on_estop)
                    
                    # Wait a moment after trigger to avoid rapid-fire events
                    time.sleep(0.5) 
                except Exception as e:
                    print(f"Keyboard listener error: {e}")
                    break # Exit listener on error or QUIT state

        # Start the listener in a separate thread
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
        print(message)  # Also print to console
    
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
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start main loop
        self.root.mainloop()
    
    def _on_closing(self):
        """Handle window close event"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.system_state.set_state(RobotState.QUIT)
            self.root.destroy()


if __name__ == "__main__":
    # Can run GUI directly
    gui = RobotBartenderGUI()
    gui.run()