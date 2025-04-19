
# -*- coding: utf-8 -*-
# Complete Training Script with Phased Learning and Auto-Drop Check (v4 - DQN + Reward Shaping)

# <<< Standard Libraries >>>
import numpy as np
import socket
import cv2
import subprocess
import os
import shutil
import glob
import imageio # Keep for potential GIF generation
import time
import traceback
import sys # Potentially useful for flushing output

# <<< Machine Learning Libraries >>>
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN # Changed from PPO to DQN
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor # Import Monitor explicitly
import torch
from stable_baselines3.common.buffers import ReplayBuffer # DQN uses ReplayBuffer

# <<< Visualization/Interaction (Optional but included from original) >>>
try:
    from IPython.display import FileLink, display
    ipython_available = True
except ImportError:
    ipython_available = False
    # Define dummy functions if not in IPython
    def display(x): pass
    def FileLink(x): return x

try:
    import pygame
    pygame_available = True
except ImportError:
    pygame_available = False
    print("Warning: Pygame module not found. Human rendering disabled.")

# --- Wandb Setup ---
try:
    from kaggle_secrets import UserSecretsClient
    secrets_available = True
except ImportError:
    secrets_available = False
    print("Kaggle secrets not available.")

try:
    import wandb
    from wandb import Settings # Import Settings here
    from wandb.integration.sb3 import WandbCallback
    wandb_module_available = True
except ImportError:
    wandb_module_available = False
    print("Wandb module not found. Logging to Wandb disabled.")
    # Dummy callback with proper __init__ for compatibility
    class WandbCallback(BaseCallback):
        """ Dummy WandbCallback to prevent errors when wandb is not installed """
        def __init__(self, *args, **kwargs):
            # Only accept 'verbose' from kwargs to pass to BaseCallback
            allowed_kwargs = {k: v for k, v in kwargs.items() if k in ['verbose']}
            super().__init__(**allowed_kwargs)
            # Initialize any attributes expected by SB3 during callback usage
            self.model_save_path = None # Example attribute
            self.gradient_save_freq = 0 # Example attribute
            self.model_save_freq = 0 # Example attribute
            self.log_interval = -1 # Example attribute

        def _init_callback(self) -> None:
            """ Initializes the callback """
            pass # No initialization needed for dummy

        def _on_step(self)->bool:
            """ Called on each step, does nothing """
            return True

        def _on_training_end(self) -> None:
            """ Called at the end of training, does nothing """
            pass

# --- Configuration ---
STUDENT_ID = "113598065" # <<< SET YOUR STUDENT ID
TOTAL_RUNTIME_ID = f"dqn_3phase_shaped_{STUDENT_ID}_{int(time.time())}" # Add "shaped" to ID

# Define Server Constants (Used by Auto-drop Check and Env)
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10612
CMD_START = b"start\n"
CMD_MOVE_LEFT = b"move -1\n"
CMD_MOVE_RIGHT = b"move 1\n"
CMD_ROTATE_LEFT = b"rotate 0\n" # Assuming rotate 0 is left/counter-clockwise
CMD_ROTATE_RIGHT = b"rotate 1\n" # Assuming rotate 1 is right/clockwise
CMD_DROP = b"drop\n"


# ========================= PHASE 1 CONFIG (DQN + Shaping) =========================
# Focus: Basic survival, line clearing without drop. High exploration. Added Height Decrease Reward.
PHASE_1_NAME = "Phase1_DQN_Shaped_ClearLines_NoDrop"
PHASE_1_TIMESTEPS = 800000 # Example: Shorter initial phase for DQN
config_p1 = {
    "phase_name": PHASE_1_NAME, "total_timesteps": PHASE_1_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_1_NAME}",
    "policy_type": "CnnPolicy", "n_stack": 4, "student_id": STUDENT_ID,
    # DQN Specific Hyperparameters
    "learning_rate": 5e-4,       # Slightly higher initial LR
    "buffer_size": 100000,       # Smaller buffer for faster initial learning
    "learning_starts": 10000,    # Start learning after 10k steps
    "batch_size": 64,            # Larger batch size
    "tau": 1.0,                  # Standard Tau for DQN
    "gamma": 0.99,               # Discount factor
    "train_freq": (4, "step"),   # Train every 4 steps
    "gradient_steps": 1,         # Train 1 gradient step per train_freq
    "target_update_interval": 1000, # Update target network less frequently initially
    "exploration_fraction": 0.8, # Explore for 80% of phase 1 timesteps
    "exploration_final_eps": 0.1,# Higher final epsilon for continued exploration
    # --- Reward Coefficients (Shaped) ---
    "reward_line_clear_coeff": 600.0,
    "reward_height_decrease_coeff": 0.1, # <<< NEW: Small reward for dropping piece lower
    "penalty_height_increase_coeff": 0.5, # Low penalty for stacking initially
    "penalty_hole_increase_coeff": 1.0,   # Low penalty for holes initially
    "penalty_step_coeff": 0.0,            # No step penalty (height decrease reward provides incentive)
    "penalty_game_over_start_coeff": 20.0, # Fixed GO penalty in this phase
    "penalty_game_over_end_coeff": 20.0,
    "curriculum_anneal_fraction": 0.0,    # No annealing for GO penalty in P1
    "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}, # Basic line multipliers
    # Environment Setting
    "remove_drop_action": True # <<< KEY: True means 4 actions, requires auto-drop check
}

# ========================= PHASE 2 CONFIG (DQN + Shaping) =========================
# Focus: Introduce Drop, Anneal Exploration, Increase GO Penalty, Increase Stacking Penalties
PHASE_2_NAME = "Phase2_DQN_Shaped_AddDrop_AnnealExplore_AddGO"
PHASE_2_TIMESTEPS = 1200000 # Longer phase to learn drop and penalties
config_p2 = {
    "phase_name": PHASE_2_NAME, "total_timesteps": PHASE_2_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_2_NAME}",
    "policy_type": "CnnPolicy", "n_stack": 4, "student_id": STUDENT_ID,
    # DQN Specific Hyperparameters (Finetuning)
    "learning_rate": 1e-4,       # Lower LR for finetuning
    "buffer_size": 400000,       # Increase buffer size
    "learning_starts": 1000,     # Assume model already learned basic interaction
    "batch_size": 32,            # Standard batch size
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": (1, "step"),   # Train more frequently
    "gradient_steps": 1,
    "target_update_interval": 5000, # Update target network more standardly
    "exploration_fraction": 0.3, # Anneal exploration over 30% of P2 steps
    "exploration_final_eps": 0.02,# Lower final epsilon
    # --- Reward Coefficients (Shaped) ---
    "reward_line_clear_coeff": 650.0, # Slightly increase line reward
    "reward_height_decrease_coeff": 0.15, # <<< Keep rewarding height decrease
    "penalty_height_increase_coeff": 2.0, # Start penalizing height more
    "penalty_hole_increase_coeff": 5.0,   # Start penalizing holes more
    "penalty_step_coeff": 0.0,
    "penalty_game_over_start_coeff": config_p1["penalty_game_over_end_coeff"], # Start where P1 ended
    "penalty_game_over_end_coeff": 100.0, # Increase final GO penalty significantly
    "curriculum_anneal_fraction": 0.6,    # Anneal GO penalty over 60% of P2 steps
    "line_clear_multipliers": {1: 1.0, 2: 4.0, 3: 9.0, 4: 16.0}, # <<< Quadratic-like bonus
    # Environment Setting
    "remove_drop_action": False # <<< KEY: False means 5 actions, includes DROP
}

# ========================= PHASE 3 CONFIG (DQN + Shaping) =========================
# Focus: Final Finetuning, Low Exploration, Strong Penalties
PHASE_3_NAME = "Phase3_DQN_Shaped_AddStackPenalty_LowExplore"
PHASE_3_TIMESTEPS = 800000 # Final tuning phase
config_p3 = {
    "phase_name": PHASE_3_NAME, "total_timesteps": PHASE_3_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_3_NAME}",
    "policy_type": "CnnPolicy", "n_stack": 4, "student_id": STUDENT_ID,
    # DQN Specific Hyperparameters (Final Polish)
    "learning_rate": 5e-5,       # Very low LR
    "buffer_size": 500000,       # Keep large buffer
    "learning_starts": 1000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": (1, "step"),
    "gradient_steps": 1,
    "target_update_interval": 10000,# Update target less frequently for stability
    "exploration_fraction": 0.05, # Very short final exploration anneal
    "exploration_final_eps": 0.01,# Very low final epsilon (almost deterministic)
    # --- Reward Coefficients (Shaped) ---
    "reward_line_clear_coeff": 700.0, # Maximize line clear reward
    "reward_height_decrease_coeff": 0.2, # <<< Slightly increase height decrease reward
    "penalty_height_increase_coeff": 7.5, # Strong height penalty (from original DQN script)
    "penalty_hole_increase_coeff": 12.5,  # Strong hole penalty (from original DQN script)
    "penalty_step_coeff": 0.0,
    "penalty_game_over_start_coeff": config_p2["penalty_game_over_end_coeff"], # Fixed high GO penalty
    "penalty_game_over_end_coeff": config_p2["penalty_game_over_end_coeff"],
    "curriculum_anneal_fraction": 0.0,    # No annealing for GO penalty in P3
    "line_clear_multipliers": {1: 1.0, 2: 4.0, 3: 9.0, 4: 25.0}, # <<< Strong Tetris bonus
    # Environment Setting
    "remove_drop_action": False # <<< KEY: False means 5 actions
}

# --- Wandb Login ---
wandb_enabled = False
WANDB_API_KEY = None
if wandb_module_available:
    try:
        # Try Kaggle secrets first
        if secrets_available:
            print("Attempting to load WANDB_API_KEY from Kaggle Secrets...")
            user_secrets = UserSecretsClient()
            WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
            print("Secret loaded from Kaggle.")
        # Fallback to environment variable
        elif "WANDB_API_KEY" in os.environ:
            print("Attempting to load WANDB_API_KEY from environment variable...")
            WANDB_API_KEY = os.environ["WANDB_API_KEY"]
            print("Secret loaded from environment variable.")
         # Try OS environment variable (if running locally)
        elif os.getenv("WANDB_API_KEY"):
             print("Attempting to load WANDB_API_KEY from OS environment...")
             WANDB_API_KEY = os.getenv("WANDB_API_KEY")
             print("Secret loaded from OS environment.")

        if WANDB_API_KEY:
            print(f"Attempting to login to Wandb (key ending with ...{WANDB_API_KEY[-4:]})...")
            try:
                # Use a reasonable timeout
                wandb.login(key=WANDB_API_KEY, timeout=45)
                wandb_enabled = True
                print("✅ Wandb login successful.")
            except Exception as login_e:
                print(f"Wandb login attempt failed: {login_e}. Running without Wandb logging.")
                wandb_enabled = False
                WANDB_API_KEY = None # Clear key on failure
        else:
            print("WANDB_API_KEY not found in Kaggle Secrets or environment variables. Running without Wandb logging.")

    except Exception as e:
        print(f"Wandb setup failed during secret retrieval/login: {e}. Running without Wandb logging.")
        wandb_enabled = False
        WANDB_API_KEY = None
else:
    print("Wandb module not installed, skipping Wandb setup.")


# --- Wandb Init (Overall Run) ---
run = None
# Ensure entity is specified if using non-default team/user
# wandb_entity = "your_wandb_entity" # <<< CHANGE THIS if needed
wandb_entity = "t113598065-ntut-edu-tw" # <<< Or use the one from the original script
project_name = f"tetris-phased-training-dqn-shaped-{STUDENT_ID}" # Specify DQN+Shaped in project
if wandb_enabled:
    try:
        run = wandb.init(
            project=project_name,
            entity=wandb_entity,
            id=TOTAL_RUNTIME_ID, # Use the defined run ID for resuming
            name=f"Run_{TOTAL_RUNTIME_ID}", # Descriptive name
            sync_tensorboard=True, # Capture SB3 TensorBoard logs
            monitor_gym=True,      # Automatically log Gym env stats
            save_code=True,        # Save main script to Wandb
            settings=Settings(init_timeout=180, start_method="thread"), # Increase timeout, use thread start method
            config={ # Log all phase configs
                "General": {"StudentID": STUDENT_ID, "RunID": TOTAL_RUNTIME_ID, "Algorithm": "DQN", "RewardShaping": True},
                "Phase1": config_p1,
                "Phase2": config_p2,
                "Phase3": config_p3
            },
            resume="allow" # Allow resuming if run ID exists
        )
        print(f"✅ Wandb run initialized/resumed. Run ID: {run.id if run else 'N/A'}")
        print(f"   View Run: {run.get_url() if run else 'N/A'}")
    except Exception as e:
        print(f"Wandb init failed: {e}.")
        wandb_enabled = False
        run = None # Ensure run is None if init fails


# --- Log & File Paths ---
# Use /kaggle/working/ for Kaggle environments, adapt if running elsewhere
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, f"tetris_train_log_{TOTAL_RUNTIME_ID}.txt")
# Temporary paths for intermediate results
phase1_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_dqn_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.zip")
phase1_stats_save_path = os.path.join(output_dir, f"vecnormalize_stats_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl")
phase2_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_dqn_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.zip")
phase2_stats_save_path = os.path.join(output_dir, f"vecnormalize_stats_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl")
# Final paths for Phase 3 results
phase3_final_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_dqn_{PHASE_3_NAME}_final_{TOTAL_RUNTIME_ID}.zip")
phase3_final_stats_save_path = os.path.join(output_dir, f"vecnormalize_stats_{PHASE_3_NAME}_final_{TOTAL_RUNTIME_ID}.pkl")


# --- Helper Functions ---

def write_log(message, exc_info=False):
    """ Writes a message to the log file and prints to console. """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    print(log_message)
    sys.stdout.flush() # Ensure message appears immediately in console/notebook output
    if exc_info:
        # Get traceback string only if exc_info is True
        log_message += "\n" + traceback.format_exc()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        # Print logging errors to console if file writing fails
        print(f"{timestamp} - Log write error: {e}")
        sys.stdout.flush()

def wait_for_tetris_server(ip=SERVER_IP, port=SERVER_PORT, timeout=60):
    """ Waits for the Tetris Java server to become available. """
    write_log(f"⏳ Waiting for Tetris server @ {ip}:{port} (timeout: {timeout}s)...")
    t_start = time.time()
    while True:
        try:
            with socket.create_connection((ip, port), timeout=1.0) as s:
                pass # Connection successful
            write_log("✅ Java server detected and ready.")
            return True
        except socket.error:
            if time.time() - t_start > timeout:
                write_log(f"❌ Server connection timed out after {timeout} seconds.")
                return False
            time.sleep(1.0) # Wait before retrying
        except Exception as e:
             write_log(f"❌ Unexpected error while waiting for server: {e}")
             return False # Treat unexpected errors as failure


# Global variable for the Java server process - MUST BE DEFINED BEFORE start_java_server
java_process = None

def start_java_server():
    """ Starts the Java Tetris server process. """
    global java_process # Allow modification of the global variable
    write_log("🚀 Attempting to start Java Tetris server...")
    # --- !!! ADAPT JAR PATH IF NEEDED !!! ---
    jar_file = "TetrisTCPserver_v0.6.jar" # Assumes JAR is in the current directory

    # Check if JAR file exists
    if not os.path.exists(jar_file):
         # Try finding it in a common input directory structure (e.g., Kaggle)
         alt_path = "/kaggle/input/tetris-server-jar/" + jar_file
         if os.path.exists(alt_path):
             jar_file = alt_path
             write_log(f"  Found JAR file at: {jar_file}")
         else:
             write_log(f"❌ CRITICAL: JAR file '{jar_file}' not found in current directory or /kaggle/input/.")
             return False # Cannot proceed without the server JAR

    try:
        # Start the Java process, redirecting stdout/stderr to prevent console clutter
        java_process = subprocess.Popen(
            ["java", "-jar", jar_file],
            stdout=subprocess.DEVNULL, # Redirect standard output
            stderr=subprocess.DEVNULL  # Redirect standard error
        )
        write_log(f"✅ Java server process initiated (PID: {java_process.pid})")

        # Wait for the server to become available
        if not wait_for_tetris_server():
            write_log("❌ Java server process started but did not become available.")
            # Attempt to terminate the unresponsive process
            if java_process and java_process.poll() is None:
                 java_process.terminate()
                 try: java_process.wait(timeout=2)
                 except subprocess.TimeoutExpired: java_process.kill()
            return False # Indicate failure

        return True # Indicate successful start and availability

    except FileNotFoundError:
        write_log("❌ Error: 'java' command not found. Is Java installed and in the system PATH?")
        return False
    except Exception as e:
        write_log(f"❌ An unexpected error occurred during Java server startup: {e}", True)
        # Ensure process termination if startup fails critically
        if java_process and java_process.poll() is None:
            java_process.terminate()
            try: java_process.wait(timeout=2)
            except subprocess.TimeoutExpired: java_process.kill()
        return False # Indicate failure


def check_server_autodrop(ip=SERVER_IP, port=SERVER_PORT, steps_to_check=30, non_drop_cmds=[CMD_MOVE_LEFT, CMD_ROTATE_RIGHT]):
    """
    Checks if the server seems to have an auto-drop mechanism. [Unchanged]
    """
    write_log("🧪 Attempting to check server for auto-drop mechanism...")
    test_sock = None
    check_successful = False
    try:
        # --- Internal Socket Helpers ---
        def receive_data(sock, size):
            data = b""
            sock.settimeout(6.0)
            t_start_recv = time.time()
            while len(data) < size:
                if time.time() - t_start_recv > 6.0:
                    raise socket.timeout(f"Timeout receiving {size} bytes (got {len(data)})")
                try:
                    chunk = sock.recv(size - len(data))
                    if not chunk: raise ConnectionAbortedError("Socket broken during check receive")
                    data += chunk
                except socket.timeout: continue
                except socket.error as recv_e: raise ConnectionAbortedError(f"Socket error during check receive: {recv_e}")
            return data
        def receive_int(sock): return int.from_bytes(receive_data(sock, 4), 'big')
        def receive_byte(sock): return receive_data(sock, 1)
        # -----------------------------
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(5.0)
        test_sock.connect((ip, port))
        write_log("  Check: Connected to server.")
        test_sock.sendall(CMD_START)
        write_log("  Check: Sent START command.")
        initial_height = -1
        try:
            is_go_byte = receive_byte(test_sock)
            lines = receive_int(test_sock)
            height = receive_int(test_sock)
            holes = receive_int(test_sock)
            img_size = receive_int(test_sock)
            if 0 < img_size <= 2000000: receive_data(test_sock, img_size)
            else: write_log(f"  Check: Warning - Invalid initial image size {img_size}")
            initial_height = height
            is_game_over = (is_go_byte == b'\x01')
            write_log(f"  Check: Initial state received (H={height}, L={lines}, Game Over={is_game_over})")
            if is_game_over: write_log("  Check: Warning - Game started in 'Game Over' state?")
        except (ConnectionAbortedError, ConnectionError, socket.timeout, ValueError) as e:
            write_log(f"⚠️ Check: Error receiving initial state: {e}. Cannot reliably verify auto-drop.")
            return False
        except Exception as e:
            write_log(f"❌ Check: Unexpected error receiving initial state: {e}", True)
            return False
        height_changed_without_drop = False
        for i in range(steps_to_check):
            command = non_drop_cmds[i % len(non_drop_cmds)]
            test_sock.sendall(command)
            time.sleep(0.2)
            try:
                is_go_byte = receive_byte(test_sock)
                lines = receive_int(test_sock)
                height = receive_int(test_sock)
                holes = receive_int(test_sock)
                img_size = receive_int(test_sock)
                if 0 < img_size <= 2000000: receive_data(test_sock, img_size)
                else: write_log(f"  Check Step {i+1}: Warning - Invalid image size {img_size}")
                height_increased = height > initial_height + 1
                game_ended_unexpectedly = (is_go_byte == b'\x01')
                if (height_increased or game_ended_unexpectedly):
                    log_reason = "Height increased" if height_increased else "Game ended"
                    write_log(f"  Check Step {i+1}: {log_reason} (H:{initial_height}->{height}, GO:{game_ended_unexpectedly}) without DROP command. Auto-drop likely present.")
                    height_changed_without_drop = True
                    break
            except (ConnectionAbortedError, ConnectionError, socket.timeout, ValueError) as e:
                write_log(f"⚠️ Check Step {i+1}: Error receiving state: {e}. Stopping check.")
                return False
            except Exception as e:
                write_log(f"❌ Check Step {i+1}: Unexpected error receiving state: {e}", True)
                return False
        check_successful = True
        if height_changed_without_drop:
            write_log("✅ Auto-drop check PASSED (Evidence of state change without DROP command found).")
            return True
        else:
            write_log(f"⚠️ Auto-drop check FAILED? (No clear evidence of auto-drop found in {steps_to_check} steps). Phase 1 (4 actions) might not work correctly.")
            return False
    except (ConnectionAbortedError, ConnectionError, socket.timeout) as e:
        write_log(f"❌ Error during auto-drop check connection/setup: {e}")
        return False
    except Exception as e:
        write_log(f"❌ Unexpected error during auto-drop check: {e}", True)
        return False
    finally:
        if test_sock:
            try: test_sock.close()
            except socket.error: pass
        if not check_successful: write_log("  Check: Auto-drop check did not complete successfully due to errors.")


# ==============================================================================
# === Tetris Environment Class Definition (Adapted for Reward Shaping) ===
# ==============================================================================
class TetrisEnv(gym.Env):
    """ Custom Gym environment for Tetris with phased config and reward shaping """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS_NO_DROP = 4
    N_DISCRETE_ACTIONS_WITH_DROP = 5
    IMG_HEIGHT = 200
    IMG_WIDTH = 100
    IMG_CHANNELS = 3
    RESIZED_DIM = 84

    def __init__(self, host_ip=SERVER_IP, host_port=SERVER_PORT, render_mode=None, env_config=None):
        super().__init__()
        self.render_mode = render_mode
        current_config = env_config
        if current_config is None: raise ValueError("env_config must be provided to TetrisEnv")

        self.server_ip = host_ip
        self.server_port = host_port
        self.config_remove_drop = current_config.get("remove_drop_action", False)

        # Define action space and command mapping based on config
        if self.config_remove_drop:
            self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS_NO_DROP)
            self.command_map = {0: CMD_MOVE_LEFT, 1: CMD_MOVE_RIGHT, 2: CMD_ROTATE_LEFT, 3: CMD_ROTATE_RIGHT}
            self._log_prefix = "[Env NoDrop]"
        else:
            self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS_WITH_DROP)
            self.command_map = {0: CMD_MOVE_LEFT, 1: CMD_MOVE_RIGHT, 2: CMD_ROTATE_LEFT, 3: CMD_ROTATE_RIGHT, 4: CMD_DROP}
            self._log_prefix = "[Env Drop]"

        # Define observation space (single grayscale image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), dtype=np.uint8)

        self.client_sock = None
        try: self._connect_socket()
        except ConnectionError as e:
             write_log(f"{self._log_prefix} ❌ Initial connection failed during __init__: {e}")
             raise ConnectionError(f"Failed to connect to Tetris server ({self.server_ip}:{self.server_port}) during environment initialization.") from e

        # Internal state variables
        self.current_cumulative_lines = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None

        # Load reward parameters from the provided configuration
        self.reward_line_clear_coeff = current_config["reward_line_clear_coeff"]
        self.reward_height_decrease_coeff = current_config.get("reward_height_decrease_coeff", 0.0) # <<< LOAD NEW COEFF
        self.penalty_height_increase_coeff = current_config["penalty_height_increase_coeff"]
        self.penalty_hole_increase_coeff = current_config["penalty_hole_increase_coeff"]
        self.penalty_step_coeff = current_config["penalty_step_coeff"]
        self.line_clear_multipliers = current_config["line_clear_multipliers"]
        self.penalty_game_over_start_coeff = current_config["penalty_game_over_start_coeff"]
        self.penalty_game_over_end_coeff = current_config["penalty_game_over_end_coeff"]
        self.current_go_penalty = self.penalty_game_over_start_coeff
        self.current_phase_name = current_config.get('phase_name', 'UnknownPhase')

        # Log initialization details
        write_log(f"{self._log_prefix} Initialized Phase: {self.current_phase_name}")
        write_log(f"{self._log_prefix} Action Space Size: {self.action_space.n}")
        write_log(f"{self._log_prefix} Rewards: LC_Base={self.reward_line_clear_coeff:.2f}, H_Decr={self.reward_height_decrease_coeff:.2f}, Step={self.penalty_step_coeff:.3f}, GO_Start={self.current_go_penalty:.2f}, GO_End={self.penalty_game_over_end_coeff:.2f}") # Log new coeff
        write_log(f"{self._log_prefix} Penalties: H_Incr={self.penalty_height_increase_coeff:.2f}, Hole={self.penalty_hole_increase_coeff:.2f}")
        write_log(f"{self._log_prefix} Line Multipliers: {self.line_clear_multipliers}")

        # Pygame related
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False
        if not pygame_available and self.render_mode == "human":
            write_log("⚠️ Pygame not available, disabling human rendering.")
            self.render_mode = None
        self._wandb_log_error_reported = False

    def set_game_over_penalty(self, new_penalty_value):
        self.current_go_penalty = new_penalty_value

    def _initialize_pygame(self):
        # [Unchanged from previous version]
        if self.render_mode == "human" and pygame_available and not self.is_pygame_initialized:
            try:
                pygame.init()
                pygame.display.init()
                self.window_surface = pygame.display.set_mode((self.RESIZED_DIM * 5, self.RESIZED_DIM * 5))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port}) - Phase: {self.current_phase_name}")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
            except Exception as e:
                write_log(f"⚠️ Pygame initialization error: {e}")
                self.render_mode = None
                self.is_pygame_initialized = False

    def _connect_socket(self):
        # [Unchanged from previous version]
        try:
            if self.client_sock:
                try: self.client_sock.shutdown(socket.SHUT_RDWR); self.client_sock.close()
                except socket.error: pass
                self.client_sock = None
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                if sys.platform == 'linux':
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            except (AttributeError, OSError): pass
            self.client_sock.settimeout(15.0)
            self.client_sock.connect((self.server_ip, self.server_port))
            self.client_sock.settimeout(15.0)
        except socket.error as e:
            write_log(f"{self._log_prefix} 🔌 Socket connection/setup error: {e}")
            raise ConnectionError(f"Failed to connect to Tetris server {self.server_ip}:{self.server_port}: {e}")

    def _send_command(self, command: bytes):
        # [Unchanged from previous version]
        if not self.client_sock:
            try: self._connect_socket()
            except ConnectionError: raise ConnectionError(f"{self._log_prefix} Socket not connected/reconnect failed for send.")
        try:
            bytes_sent = self.client_sock.sendall(command)
            if bytes_sent is not None: write_log(f"⚠️ {self._log_prefix} sendall() unexpected return: {bytes_sent}")
        except socket.timeout: raise ConnectionAbortedError(f"Send command timed out: {command.strip()}")
        except socket.error as e:
            if self.client_sock: try: self.client_sock.close(); self.client_sock = None; except socket.error: pass
            raise ConnectionAbortedError(f"Socket error sending command {command.strip()}: {e}")
        except Exception as e: raise ConnectionAbortedError(f"Unexpected error sending command: {e}")

    def _receive_data(self, size: int):
        # [Unchanged from previous version]
        if not self.client_sock:
            try: self._connect_socket()
            except ConnectionError: raise ConnectionError(f"{self._log_prefix} Socket not connected/reconnect failed for receive.")
        data = b""
        self.client_sock.settimeout(15.0)
        t_start = time.time()
        while len(data) < size:
            if time.time() - t_start > 15.0: raise socket.timeout(f"Timeout receiving {size} bytes (got {len(data)})")
            try:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    if self.client_sock: try: self.client_sock.close(); self.client_sock = None; except socket.error: pass
                    raise ConnectionAbortedError("Socket connection broken by server.")
                data += chunk
            except socket.timeout: time.sleep(0.01); continue
            except socket.error as e:
                if self.client_sock: try: self.client_sock.close(); self.client_sock = None; except socket.error: pass
                raise ConnectionAbortedError(f"Socket error receiving data: {e}")
            except Exception as e: raise ConnectionAbortedError(f"Unexpected error receiving data: {e}")
        return data

    def get_tetris_server_response(self):
        # [Unchanged from previous version - image processing and parsing]
        try:
            term_byte = self._receive_data(1)
            terminated = (term_byte == b'\x01')
            lines_cleared = int.from_bytes(self._receive_data(4), 'big')
            current_height = int.from_bytes(self._receive_data(4), 'big')
            current_holes = int.from_bytes(self._receive_data(4), 'big')
            image_size = int.from_bytes(self._receive_data(4), 'big')
            max_expected_size = self.IMG_HEIGHT * self.IMG_WIDTH * self.IMG_CHANNELS * 2
            if not 0 < image_size <= max_expected_size:
                write_log(f"{self._log_prefix} ❌ Invalid image size: {image_size}. Ending ep.")
                return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
            img_data = self._receive_data(image_size)
            nparr = np.frombuffer(img_data, np.uint8)
            np_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if np_image_bgr is None:
                write_log(f"{self._log_prefix} ❌ Image decode failed. Ending ep.")
                return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
            resized_bgr = cv2.resize(np_image_bgr, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            self.last_raw_render_frame = resized_bgr.copy()
            grayscale_obs_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(grayscale_obs_frame, axis=0).astype(np.uint8)
            self.last_observation = observation.copy()
            return terminated, lines_cleared, current_height, current_holes, observation
        except (ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError, ConnectionError, ValueError, socket.timeout) as e:
            write_log(f"{self._log_prefix} ❌ Network/Value/Timeout error getting response: {e}. Ending ep.")
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"{self._log_prefix} ❌ Unexpected error getting server response: {e}", True)
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action):
        """ Executes action, gets state, calculates SHAPED reward. """
        act_val = action.item() if isinstance(action, (np.ndarray, np.int_)) else int(action)
        command = self.command_map.get(act_val)
        if command is None:
             write_log(f"⚠️ {self._log_prefix} Invalid action: {act_val}. Sending NOP.")
             command = CMD_ROTATE_LEFT

        # Store state *before* taking the action to calculate changes
        prev_height = self.current_height
        prev_holes = self.current_holes
        prev_lines = self.current_cumulative_lines

        try:
            self._send_command(command)
            terminated, server_lines, server_height, server_holes, observation = self.get_tetris_server_response()
        except (ConnectionAbortedError, ConnectionError, ValueError, socket.timeout) as e:
            write_log(f"{self._log_prefix} ❌ Step comm/value error: {e}. Ending ep.")
            reward = -self.current_go_penalty
            info = {'lines': prev_lines, 'l': self.lifetime, 'status': 'error', 'final_status': 'comm_error'}
            log_dict_fail = { "reward_step": reward, "lines_cleared_step": 0, "height": prev_height, "holes": prev_holes, "lifetime": self.lifetime, "reward_comp/line_clear": 0.0, "reward_comp/step": 0.0, "reward_comp/height_decrease": 0.0, "reward_comp/height_penalty": 0.0, "reward_comp/hole_penalty": 0.0, "reward_comp/game_over_penalty": -self.current_go_penalty, "penalty_coeffs/game_over": self.current_go_penalty }
            self._safe_wandb_log(log_dict_fail)
            return self.last_observation.copy(), reward, True, False, info

        # --- Calculate Reward Components (SHAPED) ---
        # Note: Height/Hole changes calculated based on state *before* this step vs state *after* this step.
        lines_cleared_this_step = max(0, server_lines - prev_lines)
        height_change = server_height - prev_height
        hole_change = server_holes - prev_holes

        # 1. Line Clear Reward
        line_clear_reward = 0.0
        multiplier_used = 0.0
        if lines_cleared_this_step > 0:
            multiplier_used = self.line_clear_multipliers.get(lines_cleared_this_step, self.line_clear_multipliers.get(4, 8.0))
            line_clear_reward = multiplier_used * self.reward_line_clear_coeff

        # 2. Height Decrease Reward (Dense Shaping) <<< NEW
        # Reward if height decreased (negative height_change)
        height_decrease = max(0, -height_change)
        height_decrease_reward = height_decrease * self.reward_height_decrease_coeff

        # 3. Height Increase Penalty
        height_increase = max(0, height_change)
        height_penalty = height_increase * self.penalty_height_increase_coeff

        # 4. Hole Increase Penalty
        hole_increase = max(0, hole_change)
        hole_penalty = hole_increase * self.penalty_hole_increase_coeff

        # 5. Step Reward/Penalty
        step_reward = self.penalty_step_coeff # Can be positive or negative

        # 6. Game Over Penalty
        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.current_go_penalty

        # --- Total Reward ---
        reward = (line_clear_reward + height_decrease_reward + step_reward) \
                 - (height_penalty + hole_penalty + game_over_penalty)

        # --- Update Internal State ---
        self.current_cumulative_lines = server_lines
        self.current_height = server_height
        self.current_holes = server_holes
        self.lifetime += 1

        # --- Logging ---
        if terminated:
             write_log(f"{self._log_prefix} 💔 GameOver L={server_lines} Steps={self.lifetime} | "
                       f"RComp: LC(x{multiplier_used:.1f})={line_clear_reward:.1f} HDec={height_decrease_reward:.1f} Step={step_reward:.2f} HIncr={-height_penalty:.1f} OIncr={-hole_penalty:.1f} GO={-game_over_penalty:.1f} "
                       f"--> StepRew={reward:.2f}")
        elif lines_cleared_this_step > 0: # Log line clears
             write_log(f"{self._log_prefix} Step {self.lifetime} Lines={lines_cleared_this_step} R={reward:.2f} (LC={line_clear_reward:.1f} HDec={height_decrease_reward:.1f} HP={-height_penalty:.1f} OP={-hole_penalty:.1f})")


        # --- Wandb Logging ---
        log_dict = {
            "reward_step": reward, "lines_cleared_step": lines_cleared_this_step,
            "height": server_height, "holes": server_holes, "lifetime": self.lifetime,
            "reward_comp/line_clear": line_clear_reward,
            "reward_comp/height_decrease": height_decrease_reward, # <<< Log new component
            "reward_comp/step": step_reward,
            "reward_comp/height_penalty": -height_penalty,
            "reward_comp/hole_penalty": -hole_penalty,
            "reward_comp/game_over_penalty": -game_over_penalty,
            "penalty_coeffs/game_over": self.current_go_penalty
        }
        self._safe_wandb_log(log_dict)

        # --- Prepare Return ---
        info = { 'lines': self.current_cumulative_lines, 'l': self.lifetime, 'height': server_height, 'holes': server_holes }
        if terminated:
            info['terminal_observation'] = observation.copy()
            info['final_status'] = 'game_over'
        truncated = False

        if self.render_mode == "human": self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # [Unchanged from previous version]
        super().reset(seed=seed)
        write_log(f"{self._log_prefix} Resetting environment...")
        self._wandb_log_error_reported = False
        for attempt in range(3):
            try:
                self._connect_socket()
                self._send_command(CMD_START)
                terminated, start_lines, start_height, start_holes, initial_observation = self.get_tetris_server_response()
                if terminated or start_lines != 0:
                    write_log(f"{self._log_prefix} ⚠️ Invalid reset state (Term={terminated}, Lines={start_lines}) attempt {attempt + 1}. Retrying...")
                    if self.client_sock: try: self.client_sock.close(); self.client_sock=None; except socket.error: pass
                    time.sleep(0.5 + attempt * 0.5)
                    continue
                self.current_cumulative_lines = 0
                self.current_height = start_height
                self.current_holes = start_holes
                self.lifetime = 0
                self.last_observation = initial_observation.copy()
                self.last_raw_render_frame = None
                info = {'start_height': start_height, 'start_holes': start_holes}
                write_log(f"{self._log_prefix} Reset successful. Initial H={start_height}, O={start_holes}")
                return initial_observation, info
            except (ConnectionAbortedError, ConnectionError, ConnectionRefusedError, socket.error, TimeoutError, ValueError) as e:
                write_log(f"{self._log_prefix} 🔌 Reset conn/value error attempt {attempt + 1}/{3}: {e}")
                if self.client_sock: try: self.client_sock.close(); self.client_sock=None; except socket.error: pass
                if attempt == 2: raise RuntimeError(f"Failed reset after multiple connection attempts: {e}") from e
                time.sleep(1.0 + attempt * 0.5)
            except Exception as e:
                write_log(f"{self._log_prefix} ❌ Unexpected reset error attempt {attempt + 1}/{3}: {e}", True)
                if attempt == 2: raise RuntimeError(f"Failed reset due to unexpected error: {e}") from e
                time.sleep(1.0 + attempt * 0.5)
        write_log("❌ CRITICAL: Failed to reset environment after retry loop.")
        raise RuntimeError("Failed to reset environment after retry loop.")

    def render(self):
        # [Unchanged from previous version]
        self._initialize_pygame()
        if self.render_mode == "human" and self.is_pygame_initialized:
            if self.window_surface is None: return
            frame = self.last_raw_render_frame
            if frame is not None and frame.shape == (self.RESIZED_DIM, self.RESIZED_DIM, 3):
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    pygame.surfarray.blit_array(surf, np.transpose(rgb_frame, (1, 0, 2)))
                    scaled_surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(scaled_surf, (0, 0))
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            write_log("Pygame window closed by user."); self.close(); return
                    pygame.display.flip()
                    self.clock.tick(self.metadata["render_fps"])
                except Exception as e: write_log(f"⚠️ Pygame rendering error: {e}")
            else:
                try: self.window_surface.fill((0, 0, 0)); pygame.display.flip()
                except Exception as e: write_log(f"⚠️ Pygame fill error: {e}")
        elif self.render_mode == "rgb_array":
            if self.last_raw_render_frame is not None and self.last_raw_render_frame.shape == (self.RESIZED_DIM, self.RESIZED_DIM, 3):
                return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
            else: return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self):
        # [Unchanged from previous version]
        write_log(f"{self._log_prefix} Closing environment resources...")
        if self.client_sock:
            try: self.client_sock.shutdown(socket.SHUT_RDWR); self.client_sock.close()
            except socket.error as e: write_log(f"  Socket close error: {e}")
            finally: self.client_sock = None
        if self.is_pygame_initialized:
            try:
                if pygame_available: pygame.display.quit(); pygame.quit()
            except Exception as e: write_log(f"  Pygame close error: {e}")
            finally: self.is_pygame_initialized = False

    def _safe_wandb_log(self, data):
        # [Unchanged from previous version]
        if wandb_enabled and run:
            try:
                if wandb.run and wandb.run.id == run.id:
                    prefixed_data = {f"{self.current_phase_name}/{k}": v for k, v in data.items()}
                    wandb.log(prefixed_data, commit=False)
            except Exception as e:
                if not self._wandb_log_error_reported:
                    write_log(f"⚠️ {self._log_prefix} Wandb logging error in phase '{self.current_phase_name}': {e}")
                    self._wandb_log_error_reported = True


# ==============================================================================
# === Curriculum Callback Definition ===
# ==============================================================================
class CurriculumCallback(BaseCallback):
    """ Callback for annealing parameters like game over penalty. [Unchanged] """
    def __init__(self, penalty_start: float, penalty_end: float, anneal_fraction: float, total_training_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.penalty_start = penalty_start
        self.penalty_end = penalty_end
        self.anneal_fraction = max(0.0, min(1.0, anneal_fraction))
        self.total_training_steps = total_training_steps
        self.anneal_timesteps = 0
        if self.anneal_fraction > 0 and self.penalty_start != self.penalty_end:
             self.anneal_timesteps = int(total_training_steps * self.anneal_fraction)
        self._annealing_finished_logged = False
        self._callback_error_logged = False
        self._env_method_error_logged = False
        if self.anneal_timesteps > 0: write_log(f"[CurricCallback] Initialized: GO Penalty anneal {penalty_start:.2f} -> {penalty_end:.2f} over {self.anneal_timesteps} steps.")
        else: write_log(f"[CurricCallback] Initialized: GO Penalty fixed at {penalty_start:.2f}.")

    def _on_step(self) -> bool:
        current_penalty = self.penalty_start
        is_annealing_active = (self.anneal_timesteps > 0)
        if is_annealing_active:
            if self.num_timesteps <= self.anneal_timesteps:
                progress = max(0.0, min(1.0, self.num_timesteps / self.anneal_timesteps))
                current_penalty = self.penalty_start + progress * (self.penalty_end - self.penalty_start)
            else:
                current_penalty = self.penalty_end
                if not self._annealing_finished_logged:
                    write_log(f"[CurricCallback] GO Penalty Annealing finished at step {self.num_timesteps}. Fixed at: {current_penalty:.2f}")
                    self._annealing_finished_logged = True
        try:
            if hasattr(self.training_env, 'env_method'): self.training_env.env_method('set_game_over_penalty', current_penalty)
            elif hasattr(self.training_env, 'set_game_over_penalty'): self.training_env.set_game_over_penalty(current_penalty)
            else:
                 if not self._env_method_error_logged:
                     write_log("⚠️ [CurricCallback] Env lacks method to set GO penalty."); self._env_method_error_logged = True
            log_trigger = (self.num_timesteps % 10000 == 0) or (self.num_timesteps == 1) or (is_annealing_active and self.num_timesteps == self.anneal_timesteps + 1 and not self._annealing_finished_logged)
            if log_trigger:
                if self.logger: self.logger.record('train/current_go_penalty_coeff', current_penalty)
                elif not self._callback_error_logged: write_log("⚠️ [CurricCallback] Logger not found."); self._callback_error_logged = True
        except Exception as e:
            if not self._callback_error_logged: write_log(f"❌ [CurricCallback] Error: {e}", exc_info=True); self._callback_error_logged = True
        return True


# ==============================================================================
# === Environment Creation Helper ===
# ==============================================================================
def make_tetris_env(env_config, seed=0):
    """ Utility function to create and wrap the Tetris environment. [Unchanged] """
    def _init():
        env = TetrisEnv(env_config=env_config, render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

# ==============================================================================
# === Global Variables and Main Execution ===
# ==============================================================================
model_p1 = model_p2 = model_p3 = None
train_env_p1 = train_env_p2 = train_env_p3 = None
phase1_success = phase2_success = phase3_success = False
autodrop_check_passed = False
if torch.cuda.is_available():
    write_log(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    write_log("⚠️ GPU not detected. Using CPU.")
    device = "cpu"

# --- Main Training Loop ---
try:
    write_log(f"\n{'='*20} Starting Training Run: {TOTAL_RUNTIME_ID} {'='*20}")
    if not start_java_server():
        write_log("❌❌ CRITICAL: Java server failed to start. Aborting script. ❌❌")
        sys.exit(1)

    phase1_uses_4_actions = config_p1.get("remove_drop_action", False)
    if phase1_uses_4_actions:
        write_log("Phase 1 uses 4 actions. Performing auto-drop check...")
        autodrop_check_passed = check_server_autodrop()
        if not autodrop_check_passed: write_log("❌ Auto-drop check FAILED/Unverified. Skipping Phase 1.")
        else: write_log("✅ Auto-drop check PASSED. Proceeding with Phase 1.")
    else:
        write_log("Phase 1 uses 5 actions. Skipping auto-drop check.")
        autodrop_check_passed = True

    # ================= PHASE 1 =================
    if autodrop_check_passed:
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_1_NAME} {'='*30}")
            # --- Env Setup ---
            write_log(f"Creating Env for {PHASE_1_NAME}...")
            train_env_p1_base = DummyVecEnv([make_tetris_env(env_config=config_p1, seed=0)])
            train_env_p1_stacked = VecFrameStack(train_env_p1_base, n_stack=config_p1["n_stack"], channels_order="first")
            train_env_p1 = VecNormalize(train_env_p1_stacked, norm_obs=False, norm_reward=True, gamma=config_p1["gamma"], clip_reward=10.0)
            action_space_size_p1 = train_env_p1.action_space.n
            write_log(f" Phase 1 Env Created. Action Space Size: {action_space_size_p1}")
            expected_actions_p1 = 4 if config_p1["remove_drop_action"] else 5
            if action_space_size_p1 != expected_actions_p1: write_log(f"⚠️ WARNING: Phase 1 Action Space Mismatch! Expected {expected_actions_p1}, Got {action_space_size_p1}")

            # --- Callbacks ---
            write_log("Setting up Callbacks for Phase 1...")
            callback_list_p1 = [CurriculumCallback(config_p1["penalty_game_over_start_coeff"], config_p1["penalty_game_over_end_coeff"], config_p1["curriculum_anneal_fraction"], config_p1["total_timesteps"], verbose=1)]
            if wandb_enabled and run:
                callback_list_p1.append(WandbCallback(gradient_save_freq=20000, log="all", verbose=0))
                write_log(" Phase 1 Callbacks: Curriculum, Wandb.")
            else: write_log(" Phase 1 Callbacks: Curriculum.")

            # --- Model Creation ---
            write_log(f"Setting up NEW DQN Model for {PHASE_1_NAME}...")
            tb_log_path_p1 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_1_NAME) if wandb_enabled else None
            model_p1 = DQN(config_p1["policy_type"], train_env_p1, verbose=1, gamma=config_p1["gamma"], learning_rate=float(config_p1["learning_rate"]), buffer_size=config_p1["buffer_size"], learning_starts=config_p1["learning_starts"], batch_size=config_p1["batch_size"], tau=config_p1["tau"], train_freq=config_p1["train_freq"], gradient_steps=config_p1["gradient_steps"], target_update_interval=config_p1["target_update_interval"], exploration_fraction=config_p1["exploration_fraction"], exploration_initial_eps=1.0, exploration_final_eps=config_p1["exploration_final_eps"], seed=42, device=device, tensorboard_log=tb_log_path_p1, policy_kwargs=dict(normalize_images=True) )
            write_log(f" DQN Model created. Device: {model_p1.device}")
            write_log(f" TB Log Path (P1): {tb_log_path_p1}")

            # --- Training ---
            write_log(f"🚀 Starting Phase 1 Training ({config_p1['total_timesteps']:,} steps)...")
            t_start_p1 = time.time()
            model_p1.learn(config_p1["total_timesteps"], callback=callback_list_p1, log_interval=10, tb_log_name=PHASE_1_NAME, reset_num_timesteps=True)
            t_end_p1 = time.time()
            write_log(f"✅ Phase 1 Training Complete! Time: {(t_end_p1 - t_start_p1) / 3600:.2f} hours")

            # --- Saving ---
            write_log(f"💾 Saving Phase 1 model: {phase1_model_save_path}")
            model_p1.save(phase1_model_save_path)
            write_log(f"💾 Saving Phase 1 VecNormalize stats: {phase1_stats_save_path}")
            train_env_p1.save(phase1_stats_save_path)
            phase1_success = True

        except Exception as e: write_log(f"❌❌❌ PHASE 1 ERROR: {e}", True); phase1_success = False
        except KeyboardInterrupt: write_log(f"🛑 Phase 1 Interrupted."); phase1_success = False
        finally:
            if train_env_p1 is not None:
                try: train_env_p1.close(); write_log(" Phase 1 Env closed.")
                except Exception as e: write_log(f" Error closing P1 Env: {e}")
    else: write_log(f"⏩ Phase 1 ({PHASE_1_NAME}) was skipped.")

    # ================= PHASE 2 =================
    if phase1_success and os.path.exists(phase1_model_save_path) and os.path.exists(phase1_stats_save_path):
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_2_NAME} {'='*30}")
            # --- Env Setup ---
            write_log(f"Creating Env for {PHASE_2_NAME}...")
            train_env_p2_base = DummyVecEnv([make_tetris_env(env_config=config_p2, seed=1)])
            train_env_p2_stacked = VecFrameStack(train_env_p2_base, n_stack=config_p2["n_stack"], channels_order="first")
            write_log(f"🔄 Loading P1 VecNormalize stats: {phase1_stats_save_path}")
            train_env_p2 = VecNormalize.load(phase1_stats_save_path, train_env_p2_stacked)
            train_env_p2.training = True
            train_env_p2.gamma = config_p2["gamma"]
            action_space_size_p2 = train_env_p2.action_space.n
            write_log(f" Phase 2 Env Created/Stats Loaded. Action Space Size: {action_space_size_p2}")
            expected_actions_p2 = 4 if config_p2["remove_drop_action"] else 5
            if action_space_size_p2 != expected_actions_p2: write_log(f"⚠️ WARNING: Phase 2 Action Space Mismatch! Expected {expected_actions_p2}, Got {action_space_size_p2}")

            # --- Callbacks ---
            write_log("Setting up Callbacks for Phase 2...")
            callback_list_p2 = [CurriculumCallback(config_p2["penalty_game_over_start_coeff"], config_p2["penalty_game_over_end_coeff"], config_p2["curriculum_anneal_fraction"], config_p2["total_timesteps"], verbose=1)]
            if wandb_enabled and run:
                callback_list_p2.append(WandbCallback(gradient_save_freq=50000, log="all", verbose=0))
                write_log(" Phase 2 Callbacks: Curriculum(Active GO), Wandb.")
            else: write_log(" Phase 2 Callbacks: Curriculum(Active GO).")

            # --- Model Loading & Adaptation ---
            write_log(f"🔄 Loading Phase 1 DQN model: {phase1_model_save_path}")
            tb_log_path_p2 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_2_NAME) if wandb_enabled else None
            model_p2 = DQN.load(phase1_model_save_path, env=train_env_p2, device=device, tensorboard_log=tb_log_path_p2)
            write_log(" Phase 1 Model loaded. Policy action space potentially adapted.")
            write_log(f" TB Log Path (P2): {tb_log_path_p2}")
            write_log(f" Updating model hyperparameters for Phase 2...")
            model_p2.learning_rate = float(config_p2["learning_rate"])
            model_p2.exploration_fraction = config_p2["exploration_fraction"]
            model_p2.exploration_final_eps = config_p2["exploration_final_eps"]
            model_p2.learning_starts = config_p2["learning_starts"]
            model_p2.target_update_interval = config_p2["target_update_interval"]
            model_p2.train_freq = config_p2["train_freq"]
            if model_p2.buffer_size != config_p2["buffer_size"]: write_log(f"⚠️ WARNING: P2 buffer_size config ({config_p2['buffer_size']}) differs from loaded ({model_p2.buffer_size}). Keeping loaded size.")
            write_log(" Model hyperparameters updated.")

            # --- Training ---
            write_log(f"🚀 Starting Phase 2 Training ({config_p2['total_timesteps']:,} steps)...")
            t_start_p2 = time.time()
            model_p2.learn(config_p2["total_timesteps"], callback=callback_list_p2, log_interval=10, tb_log_name=PHASE_2_NAME, reset_num_timesteps=False)
            t_end_p2 = time.time()
            write_log(f"✅ Phase 2 Training Complete! Time: {(t_end_p2 - t_start_p2) / 3600:.2f} hours")

            # --- Saving ---
            write_log(f"💾 Saving Phase 2 model: {phase2_model_save_path}")
            model_p2.save(phase2_model_save_path)
            write_log(f"💾 Saving Phase 2 VecNormalize stats: {phase2_stats_save_path}")
            train_env_p2.save(phase2_stats_save_path)
            phase2_success = True

        except Exception as e: write_log(f"❌❌❌ PHASE 2 ERROR: {e}", True); phase2_success = False
        except KeyboardInterrupt: write_log(f"🛑 Phase 2 Interrupted."); phase2_success = False
        finally:
             if train_env_p2 is not None:
                try: train_env_p2.close(); write_log(" Phase 2 Env closed.")
                except Exception as e: write_log(f" Error closing P2 Env: {e}")
    else:
        if not phase1_success: write_log(f"\n⏩ Skipping Phase 2 ({PHASE_2_NAME}) as P1 was unsuccessful/skipped.")

    # ================= PHASE 3 =================
    if phase2_success and os.path.exists(phase2_model_save_path) and os.path.exists(phase2_stats_save_path):
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_3_NAME} {'='*30}")
            # --- Env Setup ---
            write_log(f"Creating Env for {PHASE_3_NAME}...")
            train_env_p3_base = DummyVecEnv([make_tetris_env(env_config=config_p3, seed=2)])
            train_env_p3_stacked = VecFrameStack(train_env_p3_base, n_stack=config_p3["n_stack"], channels_order="first")
            write_log(f"🔄 Loading P2 VecNormalize stats: {phase2_stats_save_path}")
            train_env_p3 = VecNormalize.load(phase2_stats_save_path, train_env_p3_stacked)
            train_env_p3.training = True
            train_env_p3.gamma = config_p3["gamma"]
            action_space_size_p3 = train_env_p3.action_space.n
            write_log(f" Phase 3 Env Created/Stats Loaded. Action Space Size: {action_space_size_p3}")
            expected_actions_p3 = 4 if config_p3["remove_drop_action"] else 5
            if action_space_size_p3 != expected_actions_p3: write_log(f"⚠️ WARNING: Phase 3 Action Space Mismatch! Expected {expected_actions_p3}, Got {action_space_size_p3}")

            # --- Callbacks ---
            write_log("Setting up Callbacks for Phase 3...")
            callback_list_p3 = [CurriculumCallback(config_p3["penalty_game_over_start_coeff"], config_p3["penalty_game_over_end_coeff"], config_p3["curriculum_anneal_fraction"], config_p3["total_timesteps"], verbose=1)]
            if wandb_enabled and run:
                callback_list_p3.append(WandbCallback(gradient_save_freq=100000, log="all", verbose=0))
                write_log(" Phase 3 Callbacks: Curriculum(Inactive GO), Wandb.")
            else: write_log(" Phase 3 Callbacks: Curriculum(Inactive GO).")

            # --- Model Loading & Adaptation ---
            write_log(f"🔄 Loading Phase 2 DQN model: {phase2_model_save_path}")
            tb_log_path_p3 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_3_NAME) if wandb_enabled else None
            model_p3 = DQN.load(phase2_model_save_path, env=train_env_p3, device=device, tensorboard_log=tb_log_path_p3)
            write_log(" Phase 2 Model loaded.")
            write_log(f" TB Log Path (P3): {tb_log_path_p3}")
            write_log(f" Updating model hyperparameters for Phase 3...")
            model_p3.learning_rate = float(config_p3["learning_rate"])
            model_p3.exploration_fraction = config_p3["exploration_fraction"]
            model_p3.exploration_final_eps = config_p3["exploration_final_eps"]
            model_p3.learning_starts = config_p3["learning_starts"]
            model_p3.target_update_interval = config_p3["target_update_interval"]
            model_p3.train_freq = config_p3["train_freq"]
            if model_p3.buffer_size != config_p3["buffer_size"]: write_log(f"⚠️ WARNING: P3 buffer_size config ({config_p3['buffer_size']}) differs from loaded ({model_p3.buffer_size}). Keeping loaded size.")
            write_log(" Model hyperparameters updated.")

            # --- Training ---
            write_log(f"🚀 Starting Phase 3 Training ({config_p3['total_timesteps']:,} steps)...")
            t_start_p3 = time.time()
            model_p3.learn(config_p3["total_timesteps"], callback=callback_list_p3, log_interval=10, tb_log_name=PHASE_3_NAME, reset_num_timesteps=False)
            t_end_p3 = time.time()
            write_log(f"✅ Phase 3 Training Complete! Time: {(t_end_p3 - t_start_p3) / 3600:.2f} hours")

            # --- Saving FINAL ---
            write_log(f"💾 Saving FINAL Phase 3 model: {phase3_final_model_save_path}")
            model_p3.save(phase3_final_model_save_path)
            write_log(f"💾 Saving FINAL Phase 3 VecNormalize stats: {phase3_final_stats_save_path}")
            train_env_p3.save(phase3_final_stats_save_path)
            if ipython_available:
                write_log("Displaying final model/stats file links:")
                display(FileLink(phase3_final_model_save_path))
                display(FileLink(phase3_final_stats_save_path))
            phase3_success = True

        except Exception as e: write_log(f"❌❌❌ PHASE 3 ERROR: {e}", True); phase3_success = False
        except KeyboardInterrupt: write_log(f"🛑 Phase 3 Interrupted."); phase3_success = False
        finally:
             if train_env_p3 is not None:
                try: train_env_p3.close(); write_log(" Phase 3 Env closed.")
                except Exception as e: write_log(f" Error closing P3 Env: {e}")
    else:
        if not phase2_success and phase1_success: write_log(f"\n⏩ Skipping Phase 3 ({PHASE_3_NAME}) as P2 was unsuccessful.")

except Exception as main_e: write_log(f"💥💥💥 UNHANDLED EXCEPTION in main script: {main_e}", True)
except KeyboardInterrupt: write_log("\n🛑🛑🛑 Main script interrupted by user (Ctrl+C). 🛑🛑🛑")
finally:
    # ================= FINAL CLEANUP =================
    write_log(f"\n{'='*20} Final Cleanup & Reporting {'='*20}")
    # --- Terminate Java Server ---
    if java_process and java_process.poll() is None:
        write_log("🧹 Terminating Java server process...")
        java_process.terminate()
        try: java_process.wait(timeout=5); write_log("✅ Java server terminated gracefully.")
        except subprocess.TimeoutExpired:
            write_log("⚠️ Java server did not terminate gracefully, killing..."); java_process.kill()
            try: java_process.wait(timeout=2); write_log("✅ Java server killed.")
            except Exception as kill_e: write_log(f"⚠️ Error waiting for killed Java process: {kill_e}")
        except Exception as e: write_log(f"⚠️ Error during Java server termination: {e}")
    elif java_process: write_log("🧹 Java server process already terminated.")
    else: write_log("🧹 Java server process not started/failed early.")

    # --- Upload Final Artifacts ---
    final_model_to_upload = None
    final_stats_to_upload = None
    final_phase_name = "None"
    if phase3_success and os.path.exists(phase3_final_model_save_path) and os.path.exists(phase3_final_stats_save_path):
        final_model_to_upload, final_stats_to_upload, final_phase_name = phase3_final_model_save_path, phase3_final_stats_save_path, PHASE_3_NAME
    elif phase2_success and os.path.exists(phase2_model_save_path) and os.path.exists(phase2_stats_save_path):
        final_model_to_upload, final_stats_to_upload, final_phase_name = phase2_model_save_path, phase2_stats_save_path, PHASE_2_NAME
    elif phase1_success and os.path.exists(phase1_model_save_path) and os.path.exists(phase1_stats_save_path):
        final_model_to_upload, final_stats_to_upload, final_phase_name = phase1_model_save_path, phase1_stats_save_path, PHASE_1_NAME

    if wandb_enabled and run and final_model_to_upload:
        write_log(f"☁️ Uploading final artifacts (from {final_phase_name}) to Wandb...")
        try:
            if wandb.run and wandb.run.id == run.id:
                model_artifact = wandb.Artifact(f"{STUDENT_ID}-dqn-model-{TOTAL_RUNTIME_ID}", type="model"); model_artifact.add_file(final_model_to_upload); run.log_artifact(model_artifact)
                write_log(f"  Model artifact logged: {os.path.basename(final_model_to_upload)}")
                stats_artifact = wandb.Artifact(f"{STUDENT_ID}-dqn-vecnorm-stats-{TOTAL_RUNTIME_ID}", type="normalization_stats"); stats_artifact.add_file(final_stats_to_upload); run.log_artifact(stats_artifact)
                write_log(f"  Stats artifact logged: {os.path.basename(final_stats_to_upload)}")
                write_log("✅ Wandb artifact upload successful.")
            else: write_log("⚠️ Wandb run inactive, cannot upload final artifacts.")
        except Exception as e: write_log(f"⚠️ Wandb artifact upload error: {e}", True)
    elif wandb_enabled and run: write_log("☁️ Skipping final artifact upload (no successful phase/files missing).")

    # --- Finish Wandb Run ---
    if wandb_enabled and run:
        write_log("Finishing Wandb run...")
        overall_success = phase3_success
        exit_code = 0 if overall_success else 1
        try:
            if wandb.run and wandb.run.id == run.id:
                run.summary["Phase1_Success"] = phase1_success
                run.summary["Phase2_Success"] = phase2_success
                run.summary["Phase3_Success"] = phase3_success
                run.summary["Overall_Success"] = overall_success
                run.finish(exit_code=exit_code)
                write_log(f"✅ Wandb run '{TOTAL_RUNTIME_ID}' finished (exit code {exit_code}).")
            else: write_log("⚠️ Wandb run already finished/inactive.")
        except Exception as finish_e: write_log(f"⚠️ Error finishing Wandb run: {finish_e}")

    # --- Final Status Report ---
    write_log(f"\n🏁🏁🏁 TRAINING RUN SUMMARY ({TOTAL_RUNTIME_ID}) 🏁🏁🏁")
    write_log(f"  Algorithm: DQN (Shaped Rewards)")
    write_log(f"  Phase 1 ({PHASE_1_NAME}) Success: {phase1_success}")
    write_log(f"  Phase 2 ({PHASE_2_NAME}) Success: {phase2_success}")
    write_log(f"  Phase 3 ({PHASE_3_NAME}) Success: {phase3_success}")
    final_result_path = "N/A"
    if final_model_to_upload: final_result_path = final_model_to_upload
    write_log(f"  Final Model Available: {final_result_path}")

    # --- Display Log File Link ---
    write_log("-" * 50)
    if os.path.exists(log_path):
        write_log(f"📜 Training log saved to: {log_path}")
        if ipython_available:
            try: display(FileLink(log_path))
            except Exception as display_e: write_log(f"(Could not display log file link: {display_e})")
    else: write_log("📜 Log file not found.")

    write_log("🏁 Script execution finished. 🏁")