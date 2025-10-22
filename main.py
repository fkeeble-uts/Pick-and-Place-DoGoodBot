import sys
import threading
import time
from SystemState import SystemState, RobotState, SequenceProgress
from robot_helpers import RobotController
import sequences
from math import pi
import numpy as np


def setup_robot_system(system_state):
  
    import swift
    from EnvironmentSetup import Scene
    from GlassBot import GlassBot
    from DrinkBot import DrinkBot
    from IngredientBot import IngredientBot
    from ServerBot import ServerBot 
    
    print("="*70)
    print("  ROBOT BARTENDER SYSTEM INITIALIZATION")
    print("="*70)
    
    env = swift.Swift()
    env.launch(realtime=True)
    scene = Scene(env)
    
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

    robot4 = ServerBot()
    robot4.base = scene.ROBOT_BASE_POSES["R4_SERVER"] 
    robot4.q = np.array([0,-pi/2,0,0,0,0])  
    robot4.add_to_env(env)

    robots = {
        'Robot1 (GlassBot)': robot1,
        'Robot2 (DrinkBot)': robot2,
        'Robot3 (IngredientBot)': robot3,
        'Robot4 (ServerBot)': robot4, 
    }
    
    
    # Create controller
    print("[INIT] Creating controller...")
    controller = RobotController(env, scene, system_state)
    
    print("\n System initialization complete!\n")
    
    return env, scene, controller, robots


def run_sequence_worker(controller, robots, scene, system_state, progress):
    """    
    Args:
        controller: RobotController instance
        robots: Dictionary of robot objects
        scene: Scene object
        system_state: SystemState for monitoring
        progress: SequenceProgress for resumability
    """
    robot1 = robots['Robot1 (GlassBot)']
    robot2 = robots['Robot2 (DrinkBot)']
    robot3 = robots['Robot3 (IngredientBot)']
    robot4 = robots['Robot4 (ServerBot)']
    
    print("\n[WORKER] Sequence worker thread started")
    
    while True:
        # Check for quit signal
        if system_state.state == RobotState.QUIT:
            print("\n[WORKER] Quit signal received - exiting")
            return
        
        # Execute sequences when in RUNNING state
        if system_state.state == RobotState.RUNNING:
            try:
                print(f"\n[WORKER] Executing sequences (Progress: {progress.get_status()})")
                
                # Execute sequences based on progress
                if progress.current_sequence in [None, 'R1']:
                    sequences.run_robot1_sequence(controller, robot1, scene, progress)
                    if system_state.state != RobotState.RUNNING:
                        continue
                
                if progress.current_sequence in [None, 'R1', 'R2']:
                    sequences.run_robot2_sequence(controller, robot2, scene, progress)
                    if system_state.state != RobotState.RUNNING:
                        continue
                
                if progress.current_sequence in [None, 'R1', 'R2', 'R3']:
                    sequences.run_robot3_sequence(controller, robot3, robot2, scene, progress)
                    if system_state.state != RobotState.RUNNING:
                        continue

                #if progress.current_sequence in [None, 'R1', 'R2', 'R3', 'R4']:
                #    sequences.run_robot4_sequence(controller, robot4, scene, progress)
                #    if system_state.state != RobotState.RUNNING:
                #        continue
                                    
                # All sequences complete
                if system_state.state == RobotState.RUNNING:
                    system_state.set_state(RobotState.TEACH)
                    progress.reset()
                    print("\n" + "="*70)
                    print(" BARTENDING SEQUENCE COMPLETE! ")
                    print("="*70 + "\n")
            
            except Exception as e:
                print(f"\n [WORKER] Critical error: {e}")
                import traceback
                traceback.print_exc()
                system_state.set_state(RobotState.ERROR)
                return
        
        # Small sleep to prevent CPU spinning
        time.sleep(0.1)


def run_terminal_mode():
    """Run system in terminal control mode (for testing)"""
    print("\n" + "="*70)
    print("  TERMINAL CONTROL MODE")
    print("="*70)
    
    # Initialize system
    system_state = SystemState()
    progress = SequenceProgress()
    env, scene, controller, robots = setup_robot_system(system_state)
    
    # Start worker thread
    worker_thread = threading.Thread(
        target=run_sequence_worker,
        args=(controller, robots, scene, system_state, progress),
        daemon=True
    )
    worker_thread.start()
    
    # Terminal control loop
    print("\nTerminal Commands:")
    print("  S - Start sequence")
    print("  E - Emergency stop")
    print("  D - Disarm E-STOP")
    print("  R - Resume sequence")
    print("  C - Clear progress")
    print("  Q - Quit")
    
    try:
        while True:
            state_str = system_state.state.name
            progress_str = progress.get_status()
            print(f"\n[{state_str}] Progress: {progress_str}")
            
            if system_state.state == RobotState.RUNNING:
                action = input("Command (E to stop): ").upper()
                if action == 'E':
                    system_state.set_state(RobotState.ESTOP_ACTIVE)
            
            elif system_state.state == RobotState.ESTOP_ACTIVE:
                action = input("Command (D to disarm): ").upper()
                if action == 'D':
                    system_state.set_state(RobotState.PAUSED)
            
            elif system_state.state == RobotState.PAUSED:
                action = input("Command (R to resume, C to clear): ").upper()
                if action == 'R':
                    system_state.set_state(RobotState.RUNNING)
                elif action == 'C':
                    progress.reset()
            
            elif system_state.state in [RobotState.TEACH, RobotState.ERROR]:
                action = input("Command (S to start, Q to quit): ").upper()
                if action == 'S':
                    system_state.set_state(RobotState.RUNNING)
                elif action == 'Q':
                    system_state.set_state(RobotState.QUIT)
                    break
            
            # Update visualization
            env.step(0.02)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt - shutting down...")
        system_state.set_state(RobotState.QUIT)
    
    # Wait for worker to finish
    worker_thread.join(timeout=2.0)
    print("\nShutdown complete")


def run_gui_mode():
    """Run system with GUI interface"""
    try:
        from gui import RobotBartenderGUI
        
        print("\n" + "="*70)
        print("  GUI MODE")
        print("="*70)
        
        # Initialize and run GUI
        gui = RobotBartenderGUI()
        gui.run()
    
    except ImportError:
        print("\n❌ GUI module not found. Please ensure gui.py is available.")
        print("Falling back to terminal mode...\n")
        run_terminal_mode()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ROBOT BARTENDER SYSTEM")
    print("  Multi-Robot Collaborative Bartending")
    print("="*70 + "\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "terminal"  # Default to GUI mode
    
    # Run appropriate mode
    if mode == "terminal" or mode == "test":
        run_terminal_mode()
    elif mode == "gui":
        run_gui_mode()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [gui|terminal]")
        sys.exit(1)