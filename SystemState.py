
import threading
from enum import Enum


class RobotState(Enum):
    """Operational states for the robot system"""
    TEACH = 1           # Manual control mode (jogging, teaching)
    RUNNING = 2         # Automatic sequence execution
    ESTOP_ACTIVE = 3    # Emergency stop - all motion halted
    PAUSED = 4          # Paused after E-STOP disarm (awaiting resume)
    ERROR = 5           # Non-recoverable error state
    QUIT = 99           # System shutdown


class SystemState:
    """Thread-safe global state manager for robot system"""
    
    def __init__(self):
        self._state = RobotState.TEACH
        self._lock = threading.Lock()
    
    @property
    def state(self):
        """Get current system state (thread-safe)"""
        with self._lock:
            return self._state
    
    def set_state(self, new_state: RobotState):
        """Set new system state with logging (thread-safe)"""
        with self._lock:
            old_state = self._state
            self._state = new_state
            print(f"\n{'='*60}")
            print(f"  STATE TRANSITION: {old_state.name} → {new_state.name}")
            print(f"{'='*60}\n")
    
    def is_movement_permitted(self):
        """Check if robot movement is allowed in current state"""
        return self.state in (RobotState.TEACH, RobotState.RUNNING)
    
    def is_estop_active(self):
        """Quick check for E-STOP state"""
        return self.state == RobotState.ESTOP_ACTIVE


class SequenceProgress:
    """
    Tracks execution progress through robot sequences for resumability.
    Allows sequences to be paused and resumed from checkpoints.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self.current_sequence = None      # Current sequence ID ('R1', 'R2', 'R3')
        self.current_checkpoint = 0       # Last completed checkpoint number
        self.sequence_complete = False
    
    def set_checkpoint(self, sequence_id: str, checkpoint_num: int):
        """
        Record progress at a checkpoint.
        
        Args:
            sequence_id: Sequence identifier (e.g., 'R1', 'R2', 'R3')
            checkpoint_num: Checkpoint number within sequence
        """
        with self._lock:
            self.current_sequence = sequence_id
            self.current_checkpoint = checkpoint_num
            print(f"CHECKPOINT: {sequence_id} - Step {checkpoint_num}")
    
    def should_skip_checkpoint(self, sequence_id: str, checkpoint_num: int) -> bool:
        """
        Determine if a checkpoint should be skipped (already completed).
        
        Args:
            sequence_id: Sequence to check
            checkpoint_num: Checkpoint number to check
            
        Returns:
            True if checkpoint should be skipped, False if it should execute
        """
        with self._lock:
            # If no progress recorded, don't skip
            if self.current_sequence is None:
                return False
            
            # If checking a different sequence
            if self.current_sequence != sequence_id:
                seq_order = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
                current_order = seq_order.get(self.current_sequence, -1)
                check_order = seq_order.get(sequence_id, -1)
                
                # Skip if this sequence is earlier than current progress
                return check_order < current_order
            
            # Same sequence - skip if checkpoint already completed
            return checkpoint_num <= self.current_checkpoint
    
    def reset(self):
        """Reset progress to start (for new sequence run)"""
        with self._lock:
            self.current_sequence = None
            self.current_checkpoint = 0
            self.sequence_complete = False
            print("Sequence progress RESET")
    
    def get_status(self) -> str:
        """Get human-readable progress status"""
        with self._lock:
            if self.current_sequence is None:
                return "Ready to start"
            return f"{self.current_sequence} - Checkpoint {self.current_checkpoint}"