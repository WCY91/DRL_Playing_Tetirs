# -*- coding: utf-8 -*-
# Complete Training Script with Phased Learning and Auto-Drop Check (v3 - KeyError Fix)

# <<< Standard Libraries >>>
import numpy as np
import socket
import cv2
import subprocess
import os
import shutil
import glob
import imageio # Although not explicitly used in final code, keep if needed elsewhere
import time
import traceback
import sys # Potentially useful for flushing output

# <<< Machine Learning Libraries >>>
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor # Import Monitor explicitly
import torch

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
STUDENT_ID = "113598065"
TOTAL_RUNTIME_ID = f"ppo_3phase_{STUDENT_ID}_{int(time.time())}" # Include Student ID in run ID

# Define Server Constants (Used by Auto-drop Check and Env)
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10612
CMD_START = b"start\n"
CMD_MOVE_LEFT = b"move -1\n"
CMD_MOVE_RIGHT = b"move 1\n"
CMD_ROTATE_LEFT = b"rotate 0\n" # Assuming rotate 0 is left/counter-clockwise
CMD_ROTATE_RIGHT = b"rotate 1\n" # Assuming rotate 1 is right/clockwise
CMD_DROP = b"drop\n"


# ========================= PHASE 1 CONFIG =========================
PHASE_1_NAME = "Phase1_ClearLines_NoDrop"
PHASE_1_TIMESTEPS = 3000000 # Example timesteps
config_p1 = {
    "phase_name": PHASE_1_NAME, "total_timesteps": PHASE_1_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_1_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.15, "learning_rate": 1e-4, "n_stack": 4, "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 50.0, "penalty_height_increase_coeff": 3.0, "penalty_hole_increase_coeff": 1.2,
    "penalty_step_coeff": -0.05, "penalty_game_over_start_coeff": 20.0, "penalty_game_over_end_coeff": 20.0,
    "curriculum_anneal_fraction": 0.0, "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0},
    "remove_drop_action": True # <<< KEY: True means 4 actions, requires auto-drop check
}

# ========================= PHASE 2 CONFIG =========================
PHASE_2_NAME = "Phase2_AddDrop_AddGO"
PHASE_2_TIMESTEPS = 2000000 # Example timesteps
config_p2 = {
    "phase_name": PHASE_2_NAME, "total_timesteps": PHASE_2_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_2_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.1, "learning_rate": 5e-5, "n_stack": 4, "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 100.0, "penalty_height_increase_coeff": 0.0, "penalty_hole_increase_coeff": 0.0,
    "penalty_step_coeff": 0.05, "penalty_game_over_start_coeff": 0.0, "penalty_game_over_end_coeff": 20.0,
    "curriculum_anneal_fraction": 0.5, "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0},
    "remove_drop_action": False # <<< KEY: False means 5 actions, includes DROP
}

# ========================= PHASE 3 CONFIG =========================
PHASE_3_NAME = "Phase3_AddStackPenalty"
PHASE_3_TIMESTEPS = 1500000 # Example timesteps
config_p3 = {
    "phase_name": PHASE_3_NAME, "total_timesteps": PHASE_3_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_3_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.05, "learning_rate": 1e-5, "n_stack": 4, "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 100.0,
    "penalty_height_increase_coeff": 0.5,
    "penalty_hole_increase_coeff": 1.0,
    "penalty_step_coeff": 0.05,
    "penalty_game_over_start_coeff": config_p2["penalty_game_over_end_coeff"],
    "penalty_game_over_end_coeff": config_p2["penalty_game_over_end_coeff"],
    "curriculum_anneal_fraction": 0.0,
    "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0},
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
project_name = f"tetris-phased-training-{STUDENT_ID}"
if wandb_enabled:
    try:
        run = wandb.init(
            project=project_name,
            id=TOTAL_RUNTIME_ID, # Use the defined run ID for resuming
            name=f"Run_{TOTAL_RUNTIME_ID}", # Descriptive name
            # Optional: Specify your wandb entity (username or team name)
            # entity="your_wandb_entity",
            sync_tensorboard=True, # Capture SB3 TensorBoard logs
            monitor_gym=True,      # Automatically log Gym env stats
            save_code=True,        # Save main script to Wandb
            settings=Settings(init_timeout=180, start_method="thread"), # Increase timeout, use thread start method
            config={ # Log all phase configs
                "General": {"StudentID": STUDENT_ID, "RunID": TOTAL_RUNTIME_ID},
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
phase1_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_ppo_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.zip")
phase1_stats_save_path = os.path.join(output_dir, f"vecnormalize_stats_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl")
phase2_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_ppo_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.zip")
phase2_stats_save_path = os.path.join(output_dir, f"vecnormalize_stats_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl")
# Final paths for Phase 3 results
phase3_final_model_save_path = os.path.join(output_dir, f"{STUDENT_ID}_ppo_{PHASE_3_NAME}_final_{TOTAL_RUNTIME_ID}.zip")
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

def start_java_server():
    """ Starts the Java Tetris server process. """
    global java_process # Allow modification of the global variable
    write_log("🚀 Attempting to start Java Tetris server...")
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
            # Optional: Use PIPE if you need to capture server output later
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE
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
    Checks if the server seems to have an auto-drop mechanism by sending non-drop commands
    and observing if the game state changes (height increases or game ends) without a drop command.
    Uses write_log for output. Returns True if auto-drop seems present, False otherwise or on error.
    """
    write_log("🧪 Attempting to check server for auto-drop mechanism...")
    test_sock = None
    check_successful = False # Flag to indicate if the check completed without errors

    try:
        # --- Internal Socket Helpers ---
        def receive_data(sock, size):
            data = b""
            # Slightly longer timeout for check reliability, but still limited
            sock.settimeout(6.0)
            t_start_recv = time.time()
            while len(data) < size:
                if time.time() - t_start_recv > 6.0:
                    raise socket.timeout(f"Timeout receiving {size} bytes (got {len(data)})")
                try:
                    # Non-blocking might be better, but stay with blocking for simplicity
                    chunk = sock.recv(size - len(data))
                    if not chunk: raise ConnectionAbortedError("Socket broken during check receive")
                    data += chunk
                except socket.timeout:
                    # Let outer loop handle timeout
                    continue
                except socket.error as recv_e:
                    raise ConnectionAbortedError(f"Socket error during check receive: {recv_e}")
            return data

        def receive_int(sock): return int.from_bytes(receive_data(sock, 4), 'big')
        def receive_byte(sock): return receive_data(sock, 1)
        # -----------------------------

        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(5.0) # Connection timeout
        test_sock.connect((ip, port))
        write_log("  Check: Connected to server.")

        test_sock.sendall(CMD_START)
        write_log("  Check: Sent START command.")

        # --- Receive Initial State ---
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
            # --- F-STRING FIX START ---
            is_game_over = (is_go_byte == b'\x01') # Evaluate comparison outside f-string
            write_log(f"  Check: Initial state received (H={height}, L={lines}, Game Over={is_game_over})")
            # --- F-STRING FIX END ---
            if is_game_over:
                 write_log("  Check: Warning - Game started in 'Game Over' state?")
                 # Might indicate an issue, but proceed with check anyway

        except (ConnectionAbortedError, ConnectionError, socket.timeout, ValueError) as e:
            write_log(f"⚠️ Check: Error receiving initial state: {e}. Cannot reliably verify auto-drop.")
            return False # Check failed
        except Exception as e:
            write_log(f"❌ Check: Unexpected error receiving initial state: {e}", True)
            return False # Check failed

        # --- Perform Check Steps ---
        height_changed_without_drop = False
        for i in range(steps_to_check):
            command = non_drop_cmds[i % len(non_drop_cmds)]
            # write_log(f"  Check Step {i+1}/{steps_to_check}: Sending {command.strip()}") # Verbose
            test_sock.sendall(command)
            # Slightly longer sleep to allow server processing and potential auto-drop
            time.sleep(0.2)

            try:
                is_go_byte = receive_byte(test_sock)
                lines = receive_int(test_sock)
                height = receive_int(test_sock)
                holes = receive_int(test_sock)
                img_size = receive_int(test_sock)
                if 0 < img_size <= 2000000: receive_data(test_sock, img_size)
                else: write_log(f"  Check Step {i+1}: Warning - Invalid image size {img_size}")

                # --- Logic for detecting auto-drop ---
                # 1. Did height change significantly from the *initial* height *without* game over?
                #    (Ignore small fluctuations, look for increase usually)
                height_increased = height > initial_height + 1 # Heuristic: height increased noticeably

                # 2. Did the game end without us sending a DROP command?
                game_ended_unexpectedly = (is_go_byte == b'\x01')

                if (height_increased or game_ended_unexpectedly):
                    log_reason = "Height increased" if height_increased else "Game ended"
                    write_log(f"  Check Step {i+1}: {log_reason} (H:{initial_height}->{height}, GO:{game_ended_unexpectedly}) without DROP command. Auto-drop likely present.")
                    height_changed_without_drop = True
                    break # Found evidence, no need to continue

                # Update initial_height if a piece locks (height usually resets low) - Optional refinement
                # if height < initial_height and height > 0: initial_height = height

            except (ConnectionAbortedError, ConnectionError, socket.timeout, ValueError) as e:
                write_log(f"⚠️ Check Step {i+1}: Error receiving state: {e}. Stopping check.")
                return False # Check failed
            except Exception as e:
                write_log(f"❌ Check Step {i+1}: Unexpected error receiving state: {e}", True)
                return False # Check failed

        # --- Final Verdict ---
        check_successful = True # Reached end without critical errors
        if height_changed_without_drop:
            write_log("✅ Auto-drop check PASSED (Evidence of state change without DROP command found).")
            return True
        else:
            write_log(f"⚠️ Auto-drop check FAILED? (No clear evidence of auto-drop found in {steps_to_check} steps). Phase 1 (4 actions) might not work correctly.")
            return False

    except (ConnectionAbortedError, ConnectionError, socket.timeout) as e:
        write_log(f"❌ Error during auto-drop check connection/setup: {e}")
        return False # Check failed
    except Exception as e:
        write_log(f"❌ Unexpected error during auto-drop check: {e}", True)
        return False # Check failed
    finally:
        if test_sock:
            try:
                test_sock.close()
                # write_log("  Check: Test socket closed.") # Optional log
            except socket.error: pass
        if not check_successful:
             write_log("  Check: Auto-drop check did not complete successfully due to errors.")


# ==============================================================================
# === Tetris Environment Class Definition ===
# ==============================================================================
class TetrisEnv(gym.Env):
    """ Custom Gym environment for interacting with the Tetris Java TCP server """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS_NO_DROP = 4
    N_DISCRETE_ACTIONS_WITH_DROP = 5
    IMG_HEIGHT = 200 # Original image height from server
    IMG_WIDTH = 100  # Original image width from server
    IMG_CHANNELS = 3 # Original image channels
    RESIZED_DIM = 84 # Dimension for the processed observation (RESIZED_DIM x RESIZED_DIM)

    def __init__(self, host_ip=SERVER_IP, host_port=SERVER_PORT, render_mode=None, env_config=None):
        super().__init__()
        self.render_mode = render_mode
        current_config = env_config
        if current_config is None:
            raise ValueError("env_config must be provided to TetrisEnv")

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
        # Initial connection attempt - handle potential failure during init
        try:
            self._connect_socket()
        except ConnectionError as e:
             write_log(f"{self._log_prefix} ❌ Initial connection failed during __init__: {e}")
             # Environment might be unusable, raise error or handle gracefully
             raise ConnectionError(f"Failed to connect to Tetris server ({self.server_ip}:{self.server_port}) during environment initialization.") from e


        # Internal state variables
        self.current_cumulative_lines = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None # Store last raw RGB frame for rendering

        # Load reward parameters from the provided configuration
        self.reward_line_clear_coeff = current_config["reward_line_clear_coeff"]
        self.penalty_height_increase_coeff = current_config["penalty_height_increase_coeff"]
        self.penalty_hole_increase_coeff = current_config["penalty_hole_increase_coeff"]
        self.penalty_step_coeff = current_config["penalty_step_coeff"]
        self.line_clear_multipliers = current_config["line_clear_multipliers"]
        self.penalty_game_over_start_coeff = current_config["penalty_game_over_start_coeff"]
        self.penalty_game_over_end_coeff = current_config["penalty_game_over_end_coeff"]
        self.current_go_penalty = self.penalty_game_over_start_coeff # Initial GO penalty
        self.current_phase_name = current_config.get('phase_name', 'UnknownPhase')

        # Log initialization details
        write_log(f"{self._log_prefix} Initialized Phase: {self.current_phase_name}")
        write_log(f"{self._log_prefix} Server Target: {self.server_ip}:{self.server_port}")
        write_log(f"{self._log_prefix} Action Space Size: {self.action_space.n}")
        write_log(f"{self._log_prefix} Rewards: LC_Base={self.reward_line_clear_coeff:.2f}, Step={self.penalty_step_coeff:.3f}, GO_Start={self.current_go_penalty:.2f}, GO_End={self.penalty_game_over_end_coeff:.2f}")
        write_log(f"{self._log_prefix} Penalties: Height={self.penalty_height_increase_coeff:.2f}, Hole={self.penalty_hole_increase_coeff:.2f}")
        write_log(f"{self._log_prefix} Line Multipliers: {self.line_clear_multipliers}")


        # Pygame related (only if pygame is available)
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False
        if not pygame_available:
             if self.render_mode == "human":
                 write_log("⚠️ Pygame not available, disabling human rendering.")
                 self.render_mode = None # Disable human rendering if pygame missing
        self._wandb_log_error_reported = False

    def set_game_over_penalty(self, new_penalty_value):
        """ Allows the game over penalty coefficient to be updated externally (e.g., by CurriculumCallback). """
        self.current_go_penalty = new_penalty_value
        # Optional: Log the update within the environment
        # write_log(f"{self._log_prefix} GO Penalty Coeff set to {new_penalty_value:.3f}")

    def _initialize_pygame(self):
        """ Initializes Pygame for human rendering if not already done. """
        if self.render_mode == "human" and pygame_available and not self.is_pygame_initialized:
            try:
                pygame.init()
                pygame.display.init()
                 # Create a reasonably sized window
                self.window_surface = pygame.display.set_mode((self.RESIZED_DIM * 5, self.RESIZED_DIM * 5))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port}) - Phase: {self.current_phase_name}")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log("  Pygame initialized for human rendering.")
            except Exception as e:
                write_log(f"⚠️ Pygame initialization error: {e}")
                self.render_mode = None # Disable human rendering on error
                self.is_pygame_initialized = False # Ensure flag is false

    def _connect_socket(self):
        """ Establishes or re-establishes the socket connection to the server. """
        try:
            # Close existing socket cleanly if it exists
            if self.client_sock:
                try:
                    self.client_sock.shutdown(socket.SHUT_RDWR) # Signal close intent
                    self.client_sock.close()
                except socket.error: pass # Ignore errors on close
                self.client_sock = None # Reset variable

            # Create and connect new socket
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set socket options: Keepalive and potentially LINGER
            self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # TCP Keepalive parameters (OS specific, example for Linux)
            try:
                if sys.platform == 'linux':
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30) # Idle time
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10) # Interval
                     self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3) # Probes
            except (AttributeError, OSError): # Handle if options not available
                 pass
            self.client_sock.settimeout(15.0) # Generous connection timeout
            self.client_sock.connect((self.server_ip, self.server_port))
            # Set a working timeout for subsequent operations
            self.client_sock.settimeout(15.0)
            # write_log(f"{self._log_prefix} Socket connected to {self.server_ip}:{self.server_port}") # Optional success log

        except socket.error as e:
            write_log(f"{self._log_prefix} 🔌 Socket connection/setup error: {e}")
            # Reraise as a specific error type for calling code to handle
            raise ConnectionError(f"Failed to connect to Tetris server {self.server_ip}:{self.server_port}: {e}")

    def _send_command(self, command: bytes):
        """ Sends a command to the Tetris server. """
        if not self.client_sock:
            write_log(f"{self._log_prefix} ❌ Attempted to send command but socket is not connected.")
            # Try to reconnect before failing
            try:
                 write_log(f"{self._log_prefix} Attempting reconnect before send...")
                 self._connect_socket()
            except ConnectionError:
                 raise ConnectionError(f"{self._log_prefix} Socket is not connected and reconnect failed for send.")

        try:
            bytes_sent = self.client_sock.sendall(command)
            if bytes_sent is not None: # sendall returns None on success
                 write_log(f"⚠️ {self._log_prefix} sendall() did not return None (returned {bytes_sent}). Unexpected behavior?")
            # write_log(f"Sent: {command.strip()}") # Verbose logging
        except socket.timeout:
            write_log(f"{self._log_prefix} ❌ Send command timed out: {command.strip()}")
            raise ConnectionAbortedError(f"Send command timed out: {command.strip()}")
        except socket.error as e:
            write_log(f"{self._log_prefix} ❌ Socket error sending command {command.strip()}: {e}")
             # Attempt to close the potentially broken socket before raising
            if self.client_sock:
                try: self.client_sock.close()
                except socket.error: pass
                self.client_sock = None
            raise ConnectionAbortedError(f"Socket error sending command {command.strip()}: {e}")
        except Exception as e:
             write_log(f"{self._log_prefix} ❌ Unexpected error sending command {command.strip()}: {e}", True)
             raise ConnectionAbortedError(f"Unexpected error sending command: {e}")


    def _receive_data(self, size: int):
        """ Receives a specific number of bytes from the server. """
        if not self.client_sock:
            write_log(f"{self._log_prefix} ❌ Attempted to receive data but socket is not connected.")
            # Try to reconnect
            try:
                 write_log(f"{self._log_prefix} Attempting reconnect before receive...")
                 self._connect_socket()
            except ConnectionError:
                 raise ConnectionError(f"{self._log_prefix} Socket is not connected and reconnect failed for receive.")

        data = b""
        self.client_sock.settimeout(15.0) # Ensure receive timeout is set
        t_start = time.time()
        while len(data) < size:
            # Check overall timeout for receiving this specific block of data
            if time.time() - t_start > 15.0:
                write_log(f"{self._log_prefix} ❌ Timeout receiving {size} bytes (received {len(data)}).")
                raise socket.timeout(f"Timeout receiving {size} bytes (received {len(data)})")
            try:
                # Receive remaining bytes needed in this chunk
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    # Socket closed unexpectedly by the server
                    write_log(f"{self._log_prefix} ❌ Socket connection broken by server (received empty chunk).")
                    # Close our end of the socket
                    if self.client_sock:
                        try: self.client_sock.close()
                        except socket.error: pass
                        self.client_sock = None
                    raise ConnectionAbortedError("Socket connection broken by server.")
                data += chunk
            except socket.timeout:
                # recv timed out, but overall time might still be okay. Continue loop.
                # Add a small sleep to prevent tight loop hammering CPU on repeated timeouts
                time.sleep(0.01)
                continue
            except socket.error as e:
                write_log(f"{self._log_prefix} ❌ Socket error during receive: {e}")
                 # Close potentially broken socket
                if self.client_sock:
                    try: self.client_sock.close()
                    except socket.error: pass
                    self.client_sock = None
                raise ConnectionAbortedError(f"Socket error receiving data: {e}")
            except Exception as e:
                write_log(f"{self._log_prefix} ❌ Unexpected error receiving data: {e}", True)
                raise ConnectionAbortedError(f"Unexpected error receiving data: {e}")
        return data # Return the complete data

    def get_tetris_server_response(self):
        """ Receives and parses the full state response from the Tetris server. """
        try:
            # Receive state components in order
            term_byte = self._receive_data(1)
            terminated = (term_byte == b'\x01')
            lines_cleared = int.from_bytes(self._receive_data(4), 'big')
            current_height = int.from_bytes(self._receive_data(4), 'big')
            current_holes = int.from_bytes(self._receive_data(4), 'big')
            image_size = int.from_bytes(self._receive_data(4), 'big')

            # Validate image size
            # Max reasonable size: RESIZED_DIM * RESIZED_DIM * 3 (color) * some factor for compression inefficiency
            max_expected_size = self.IMG_HEIGHT * self.IMG_WIDTH * self.IMG_CHANNELS * 2 # Generous buffer
            if not 0 < image_size <= max_expected_size:
                write_log(f"{self._log_prefix} ❌ Invalid image size received: {image_size}. Max expected: {max_expected_size}. Ending episode.")
                # Return game over state to safely terminate the episode
                return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

            # Receive image data
            img_data = self._receive_data(image_size)

            # Decode image data
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode as color image first
            np_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Check if decoding was successful
            if np_image_bgr is None:
                write_log(f"{self._log_prefix} ❌ Image decoding failed (received {len(img_data)} bytes). Using last valid observation. Ending episode.")
                # Return game over state
                return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

            # --- Image Processing ---
            # Resize the color image (important for rendering)
            resized_bgr = cv2.resize(np_image_bgr, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            # Store the resized *color* frame for potential rendering
            self.last_raw_render_frame = resized_bgr.copy()

            # Convert the *resized* image to grayscale for the observation
            grayscale_obs_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

            # Add channel dimension (required by SB3 CNN policies: C, H, W)
            observation = np.expand_dims(grayscale_obs_frame, axis=0).astype(np.uint8)

            # Store the processed observation as the last valid one
            self.last_observation = observation.copy()

            # Return parsed state
            return terminated, lines_cleared, current_height, current_holes, observation

        # Handle specific network/parsing errors
        except (ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError, ConnectionError, ValueError, socket.timeout) as e:
            write_log(f"{self._log_prefix} ❌ Network/Value/Timeout error getting response: {e}. Ending episode.")
            # Return game over state with last known values
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
        # Handle any other unexpected errors
        except Exception as e:
            write_log(f"{self._log_prefix} ❌ Unexpected error getting server response: {e}", True)
            # Return game over state
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action):
        """ Executes an action, gets the new state, calculates reward, and returns results. """
        # Ensure action is a valid integer
        act_val = action.item() if isinstance(action, (np.ndarray, np.int_)) else int(action)

        # Get the command corresponding to the action
        command = self.command_map.get(act_val)
        if command is None:
             # Handle invalid action (e.g., if action space changes unexpectedly)
             write_log(f"⚠️ {self._log_prefix} Invalid action received in step: {act_val}. Sending NOP (Rotate Left).")
             command = CMD_ROTATE_LEFT # Default to a No-Operation action
        # write_log(f"{self._log_prefix} Step {self.lifetime + 1}: Chosen Action={act_val}, Command={command.strip()}")
        # Send command and get response
        try:
            self._send_command(command)
            terminated, server_lines, server_height, server_holes, observation = self.get_tetris_server_response()
        except (ConnectionAbortedError, ConnectionError, ValueError, socket.timeout) as e:
            # Handle critical communication errors during step
            write_log(f"{self._log_prefix} ❌ Step communication/value error: {e}. Ending episode.")
            # Apply Game Over penalty if communication fails mid-episode
            reward = -self.current_go_penalty
            info = {'lines': self.current_cumulative_lines, 'l': self.lifetime, 'status': 'error', 'final_status': 'comm_error'}
            # Return state consistent with termination (terminated=True, truncated=True)
            # In Gymnasium, truncated is now separate from terminated. Communication error leads to termination.
            return self.last_observation.copy(), reward, True, False, info # terminated=True, truncated=False

        # --- Calculate Reward Components ---
        # Lines cleared in this step
        lines_cleared_this_step = max(0, server_lines - self.current_cumulative_lines)
        line_clear_reward = 0.0
        multiplier_used = 0.0
        if lines_cleared_this_step > 0:
            # Apply multiplier from config, default to highest for >4 lines
            multiplier_used = self.line_clear_multipliers.get(lines_cleared_this_step, self.line_clear_multipliers.get(4, 8.0))
            line_clear_reward = multiplier_used * self.reward_line_clear_coeff

        # Height increase penalty
        height_increase = max(0, server_height - self.current_height)
        height_penalty = height_increase * self.penalty_height_increase_coeff

        # Hole increase penalty
        hole_increase = max(0, server_holes - self.current_holes)
        hole_penalty = hole_increase * self.penalty_hole_increase_coeff

        # Step reward (survival bonus or cost per step)
        step_reward = self.penalty_step_coeff # Note: Name is penalty but often used as positive reward

        # Game Over penalty (using potentially annealed value)
        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.current_go_penalty

        # --- Total Reward ---
        reward = (line_clear_reward + step_reward) - (height_penalty + hole_penalty + game_over_penalty)

        # --- Logging Reward Breakdown (Optional but helpful) ---
        # Log details especially on game over or significant events
        log_reward_details = terminated or (lines_cleared_this_step > 0) or (self.lifetime % 100 == 0) # Example log trigger
        if log_reward_details and not terminated: # Don't log regular steps if logging GO below
            # pass # Or log less detailed info for regular steps
            if self.lifetime % 500 == 0: # Log intermediate reward details occasionally
                 write_log(f"{self._log_prefix} Step {self.lifetime+1} R={reward:.2f} (LCR={line_clear_reward:.1f}, SR={step_reward:.2f}, HP={-height_penalty:.1f}, OP={-hole_penalty:.1f})")

        if terminated:
             write_log(f"{self._log_prefix} 💔 GameOver L={server_lines} Steps={self.lifetime + 1} | "
                       f"Reward Comp: LC(x{multiplier_used:.1f})={line_clear_reward:.1f} Step={step_reward:.2f} HIncr={-height_penalty:.1f} OIncr={-hole_penalty:.1f} GO={-game_over_penalty:.1f} "
                       f"--> StepRew={reward:.2f}")


        # Update internal state *after* calculating rewards based on change
        self.current_cumulative_lines = server_lines
        self.current_height = server_height
        self.current_holes = server_holes
        self.lifetime += 1

        # Prepare info dictionary (standard for Gym)
        info = {
            'lines': self.current_cumulative_lines,
            'l': self.lifetime, # Use 'l' for length/lifetime as common in SB3 logs
            'height': server_height,
            'holes': server_holes
            }

        # ===================== KEY CHANGE HERE =====================
        if terminated:
            # Add required keys for SB3 Monitor wrapper and logging callbacks
            # Monitor wrapper looks for info['episode'] upon termination
            info['terminal_observation'] = observation.copy()
            info['episode'] = {
                'l': self.lifetime,
                'lines': self.current_cumulative_lines, # Final lines for episode
                'final_height': server_height,
                'final_holes': server_holes,
                'game_over_penalty_applied': game_over_penalty, # Log the applied penalty
                # Add placeholders for keys potentially accessed by callbacks before Monitor adds them
                'r': 0.0,  # Placeholder reward, Monitor calculates actual cumulative 'r'
                't': time.time() # Placeholder time, Monitor calculates actual 't'
            }
        # ===================== END KEY CHANGE =====================

        # Log step details for Wandb (using internal safe log method)
        log_dict = {
            "reward_step": reward, # Reward for this specific step
            "lines_cleared_step": lines_cleared_this_step,
            "height": server_height,
            "holes": server_holes,
            "lifetime": self.lifetime,
            "reward_comp/line_clear": line_clear_reward,
            "reward_comp/step": step_reward,
            "reward_comp/height_penalty": -height_penalty,
            "reward_comp/hole_penalty": -hole_penalty,
            "reward_comp/game_over_penalty": -game_over_penalty,
            "penalty_coeffs/game_over": self.current_go_penalty # Log current coeff value
        }
        self._safe_wandb_log(log_dict)

        # Render the current state if human mode is enabled
        if self.render_mode == "human":
            self.render()

        # Return standard Gym step tuple: observation, reward, terminated, truncated, info
        # Truncated is typically used for time limits, not natural game termination
        truncated = False # Game termination is handled by 'terminated'

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """ Resets the environment to a starting state. """
        # Reset seed if provided (standard Gym practice)
        super().reset(seed=seed)
        write_log(f"{self._log_prefix} Resetting environment...")
        self._wandb_log_error_reported = False # Reset wandb error flag for new episode

        # Retry loop for robustness against connection issues during reset
        for attempt in range(3):
            try:
                # Ensure connection is fresh for the new game
                self._connect_socket()
                # Send the start command
                self._send_command(CMD_START)
                # Get the initial state after starting
                terminated, start_lines, start_height, start_holes, initial_observation = self.get_tetris_server_response()

                # Check if the server reset correctly (should not be terminated, lines should be 0)
                if terminated or start_lines != 0:
                    write_log(f"{self._log_prefix} ⚠️ Invalid reset state (Term={terminated}, Lines={start_lines}) on attempt {attempt + 1}. Retrying...")
                    # Close potentially bad socket before retrying
                    if self.client_sock:
                         try: self.client_sock.close()
                         except socket.error: pass
                         self.client_sock = None
                    time.sleep(0.5 + attempt * 0.5) # Wait longer on subsequent attempts
                    continue # Go to the next attempt

                # --- Successful Reset ---
                # Reset internal environment state
                self.current_cumulative_lines = 0
                self.current_height = start_height
                self.current_holes = start_holes
                self.lifetime = 0
                self.last_observation = initial_observation.copy()
                self.last_raw_render_frame = None # Clear render frame

                # Prepare info dictionary for Gym v26 API
                info = {'start_height': start_height, 'start_holes': start_holes}
                write_log(f"{self._log_prefix} Reset successful. Initial H={start_height}, O={start_holes}")
                # Return initial observation and info dict
                return initial_observation, info

            # Handle connection errors during reset attempt
            except (ConnectionAbortedError, ConnectionError, ConnectionRefusedError, socket.error, TimeoutError, ValueError) as e:
                write_log(f"{self._log_prefix} 🔌 Reset connection/value error attempt {attempt + 1}/{3}: {e}")
                # Clean up socket before retrying
                if self.client_sock:
                    try: self.client_sock.close()
                    except socket.error: pass
                    self.client_sock = None

                if attempt == 2: # If last attempt failed
                    write_log(f"❌ CRITICAL: Failed to reset environment after {attempt + 1} attempts due to connection errors.")
                    raise RuntimeError(f"Failed to reset environment after multiple connection attempts: {e}") from e
                time.sleep(1.0 + attempt * 0.5) # Wait longer before next attempt

            # Handle other unexpected errors during reset
            except Exception as e:
                write_log(f"{self._log_prefix} ❌ Unexpected reset error attempt {attempt + 1}/{3}: {e}", True)
                if attempt == 2: # Raise error on last attempt
                    write_log(f"❌ CRITICAL: Failed to reset environment after {attempt + 1} attempts due to unexpected error.")
                    raise RuntimeError(f"Failed reset due to unexpected error: {e}") from e
                time.sleep(1.0 + attempt * 0.5)

        # This part should ideally not be reached if exceptions are raised correctly
        write_log("❌ CRITICAL: Failed to reset environment after retry loop completed without success.")
        raise RuntimeError("Failed to reset environment after retry loop.")

    def render(self):
        """ Renders the environment state. """
        # Initialize pygame if rendering in human mode and not already done
        self._initialize_pygame()

        if self.render_mode == "human" and self.is_pygame_initialized:
            if self.window_surface is None:
                 write_log("⚠️ Render called in human mode, but window surface is None.")
                 return # Cannot render

            frame = self.last_raw_render_frame # Use the stored color frame
            if frame is not None and frame.shape == (self.RESIZED_DIM, self.RESIZED_DIM, 3):
                try:
                    # Pygame expects RGB, OpenCV uses BGR by default
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Create a pygame surface from the numpy array
                    # Note: Pygame surfaces use (width, height), numpy arrays use (height, width)
                    # Transpose is needed if converting directly. Simpler to create surface and blit.
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    # Blit the array onto the surface. Transpose (1, 0, 2) converts HWC (numpy) to WHC (pygame expects W, H for blit)
                    pygame.surfarray.blit_array(surf, np.transpose(rgb_frame, (1, 0, 2)))

                    # Scale the surface to the display window size
                    scaled_surf = pygame.transform.scale(surf, self.window_surface.get_size())

                    # Draw the scaled surface onto the window
                    self.window_surface.blit(scaled_surf, (0, 0))

                    # Handle pygame events (like closing the window)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                             write_log("Pygame window closed by user.")
                             self.close() # Clean up pygame and socket
                             # Potentially signal the training loop to stop?
                             # This might require more complex handling depending on the training framework.
                             # For now, just close the env resources.
                             return # Stop rendering this frame

                    # Update the display
                    pygame.display.flip()

                    # Maintain target frame rate
                    self.clock.tick(self.metadata["render_fps"])

                except Exception as e:
                    # Log rendering errors but try to continue
                    write_log(f"⚠️ Pygame rendering error: {e}")
            else:
                # If no valid frame available, draw a black screen or placeholder
                try:
                    self.window_surface.fill((0, 0, 0)) # Black screen
                    # Optional: Add text indicating no frame
                    # font = pygame.font.Font(None, 30)
                    # text = font.render("No Frame", True, (255, 255, 255))
                    # self.window_surface.blit(text, (10, 10))
                    pygame.display.flip()
                except Exception as e:
                    write_log(f"⚠️ Pygame fill error: {e}")

        elif self.render_mode == "rgb_array":
            # Return the last captured raw frame (stored as BGR), converted to RGB
            if self.last_raw_render_frame is not None and self.last_raw_render_frame.shape == (self.RESIZED_DIM, self.RESIZED_DIM, 3):
                return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
            else:
                # Return a black frame if no valid frame has been captured yet
                return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self):
        """ Closes the socket connection and cleans up pygame resources. """
        write_log(f"{self._log_prefix} Closing environment resources...")
        # Close socket connection
        if self.client_sock:
            try:
                # Shutdown communication first
                self.client_sock.shutdown(socket.SHUT_RDWR)
                self.client_sock.close()
                write_log("  Socket closed.")
            except socket.error as e:
                 # Log error but continue cleanup
                 write_log(f"  Socket close error: {e}")
            finally:
                 self.client_sock = None # Ensure variable is reset

        # Quit pygame if it was initialized
        if self.is_pygame_initialized:
            try:
                if pygame_available: # Check again just in case
                    pygame.display.quit()
                    pygame.quit()
                    write_log("  Pygame closed.")
            except Exception as e:
                 write_log(f"  Pygame close error: {e}") # Log error during quit
            finally:
                 self.is_pygame_initialized = False # Ensure flag is reset

    def _safe_wandb_log(self, data):
        """ Safely logs data to Wandb if enabled and run is active, prefixing keys with phase name. """
        if wandb_enabled and run:
            try:
                # Check if the run is still active before logging
                # wandb.run is the currently active run
                if wandb.run and wandb.run.id == run.id:
                    # Add phase prefix to all keys
                    prefixed_data = {f"{self.current_phase_name}/{k}": v for k, v in data.items()}
                    # commit=False because SB3 handles committing logs at intervals
                    wandb.log(prefixed_data, commit=False)
                # else: # Optional: Log if trying to log to an inactive run
                #     if not self._wandb_log_error_reported:
                #         write_log(f"⚠️ {self._log_prefix} Wandb run seems inactive, skipping log.")
                #         self._wandb_log_error_reported = True
            except Exception as e:
                # Log error only once per episode to avoid spamming
                if not self._wandb_log_error_reported:
                    write_log(f"⚠️ {self._log_prefix} Wandb logging error in phase '{self.current_phase_name}': {e}")
                    self._wandb_log_error_reported = True # Set flag


# ==============================================================================
# === Curriculum Callback Definition ===
# ==============================================================================
class CurriculumCallback(BaseCallback):
    """
    A Stable Baselines3 callback to anneal the game over penalty during training.
    """
    def __init__(self, penalty_start: float, penalty_end: float, anneal_fraction: float, total_training_steps: int, verbose: int = 0):
        """
        :param penalty_start: Initial game over penalty coefficient.
        :param penalty_end: Final game over penalty coefficient.
        :param anneal_fraction: Fraction of total_training_steps over which to anneal.
        :param total_training_steps: Total timesteps for the current training phase.
        :param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.penalty_start = penalty_start
        self.penalty_end = penalty_end
        self.anneal_fraction = max(0.0, min(1.0, anneal_fraction)) # Ensure fraction is [0, 1]
        self.total_training_steps = total_training_steps
        # Calculate timesteps for annealing, 0 if fraction is 0 or start/end are same
        self.anneal_timesteps = 0
        if self.anneal_fraction > 0 and self.penalty_start != self.penalty_end:
             self.anneal_timesteps = int(total_training_steps * self.anneal_fraction)

        # Flags to prevent repeated logging messages
        self._annealing_finished_logged = False
        self._callback_error_logged = False # Tracks general callback errors
        self._env_method_error_logged = False # Tracks specific env_method errors

        # Log initialization details
        if self.anneal_timesteps > 0:
            write_log(f"[CurricCallback] Initialized: GO Penalty anneal {penalty_start:.2f} -> {penalty_end:.2f} over {self.anneal_timesteps} steps (total phase steps: {total_training_steps}).")
        else:
            write_log(f"[CurricCallback] Initialized: GO Penalty fixed at {penalty_start:.2f} (Anneal fraction: {self.anneal_fraction:.2f}).")

    def _on_step(self) -> bool:
        """
        Called by SB3 on each step. Updates the penalty in the environment(s).
        """
        # Determine the current target penalty based on annealing progress
        current_penalty = self.penalty_start
        is_annealing_active = (self.anneal_timesteps > 0)

        if is_annealing_active:
            if self.num_timesteps <= self.anneal_timesteps:
                # Calculate linear annealing progress
                progress = max(0.0, min(1.0, self.num_timesteps / self.anneal_timesteps))
                current_penalty = self.penalty_start + progress * (self.penalty_end - self.penalty_start)
            else: # Annealing period finished
                current_penalty = self.penalty_end
                # Log completion message only once
                if not self._annealing_finished_logged:
                    write_log(f"[CurricCallback] Annealing finished at step {self.num_timesteps}. GO Penalty fixed at: {current_penalty:.2f}")
                    self._annealing_finished_logged = True
        else:
             # If no annealing, penalty is always the starting value (should equal end value)
             current_penalty = self.penalty_start

        # Update the penalty in the environment(s) using VecEnv's env_method
        try:
             # Check if the training environment supports env_method (for VecEnvs)
             if hasattr(self.training_env, 'env_method') and callable(getattr(self.training_env, 'env_method')):
                 # Call the 'set_game_over_penalty' method in each sub-environment
                 # Pass current_penalty as the argument
                 results = self.training_env.env_method('set_game_over_penalty', current_penalty)
                 # Optional: Check results if needed, env_method returns a list of results from each env
             # Handle non-vectorized environments directly
             elif hasattr(self.training_env, 'set_game_over_penalty') and callable(getattr(self.training_env, 'set_game_over_penalty')):
                  self.training_env.set_game_over_penalty(current_penalty)
             else:
                  # Log error only once if the necessary methods are missing
                  if not self._env_method_error_logged:
                      write_log("⚠️ [CurricCallback] Warning: training_env does not support 'env_method' or 'set_game_over_penalty'. Cannot update penalty.")
                      self._env_method_error_logged = True

             # Log the current penalty coefficient to TensorBoard/Wandb via SB3 logger
             # Log less frequently to avoid cluttering logs
             log_trigger = (self.num_timesteps % 10000 == 0) or \
                           (self.num_timesteps == 1) or \
                           (is_annealing_active and self.num_timesteps == self.anneal_timesteps + 1 and not self._annealing_finished_logged)

             if log_trigger:
                 if self.logger: # Check if logger is available
                      self.logger.record('train/current_go_penalty_coeff', current_penalty)
                      # Optional: Print verbose log using write_log
                      # if self.verbose > 0:
                      #     write_log(f"[CurricCallback] Step {self.num_timesteps}: GO Penalty Coeff = {current_penalty:.3f}")
                 elif not self._callback_error_logged: # Log logger warning only once
                      write_log("⚠️ [CurricCallback] Logger not available. Cannot record penalty coefficient.")
                      self._callback_error_logged = True # Use general error flag

        except Exception as e:
            # Log any other errors during callback execution only once
            if not self._callback_error_logged:
                write_log(f"❌ [CurricCallback] Error during _on_step: {e}", exc_info=True)
                self._callback_error_logged = True # Prevent spamming

        return True # Must return True to continue training


# ==============================================================================
# === Environment Creation Helper ===
# ==============================================================================
# Define a helper function to create the environment and wrap it correctly
def make_tetris_env(env_config, seed=0):
    """
    Utility function for multiprocessed env. Creates and wraps the Tetris environment.

    :param env_config: Configuration dictionary for the TetrisEnv.
    :param seed: the initial seed for RNG.
    """
    def _init():
        # Pass the full env_config to the TetrisEnv
        env = TetrisEnv(env_config=env_config, render_mode=None)
        # IMPORTANT: Wrap with Monitor FIRST, before other wrappers if possible
        # Monitor keeps track of episode stats like reward ('r') and length ('l')
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

# ==============================================================================
# === Global Variables and Main Execution ===
# ==============================================================================

# Global variable for the Java server process
java_process = None

# --- Global Variables for Models & Envs (initialize to None) ---
model_p1 = model_p2 = model_p3 = None
train_env_p1 = train_env_p2 = train_env_p3 = None
phase1_success = phase2_success = phase3_success = False
autodrop_check_passed = False # Flag specifically for the check result

# --- Check GPU Availability ---
if torch.cuda.is_available():
    write_log(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    write_log("⚠️ GPU not detected. Using CPU.")
    device = "cpu"


# --- Main Training Loop ---
try:
    # === Pre-Training Setup ===
    write_log(f"\n{'='*20} Starting Training Run: {TOTAL_RUNTIME_ID} {'='*20}")

    # 1. Start Java Server
    if not start_java_server():
        # Critical failure if server doesn't start
        write_log("❌❌ CRITICAL: Java server failed to start. Aborting script. ❌❌")
        # Use sys.exit(1) for non-interactive environments or raise error
        sys.exit(1) # Exit with error code
        # raise RuntimeError("Java Server Failed to Start")

    # 2. Check for Server Auto-Drop (Conditional based on Phase 1 config)
    phase1_uses_4_actions = config_p1.get("remove_drop_action", False)

    if phase1_uses_4_actions:
        write_log("Phase 1 configured for 4 actions (remove_drop_action=True).")
        write_log("Performing server auto-drop check...")
        # Run the check function
        autodrop_check_passed = check_server_autodrop()

        if not autodrop_check_passed:
            write_log("❌ Server auto-drop check FAILED or could not be verified.")
            write_log("   Since Phase 1 requires auto-drop (4 actions), Phase 1 will be SKIPPED.")
            # No need to set model/stats path to None, phase1_success remains False
        else:
            write_log("✅ Server auto-drop check PASSED (or evidence found). Proceeding with Phase 1.")
    else:
        write_log("Phase 1 configured for 5 actions (remove_drop_action=False).")
        write_log("Skipping server auto-drop check as it's not required.")
        autodrop_check_passed = True # Set to True because the check is not needed for 5 actions


    # ==============================================================================
    # === PHASE 1: Initial Training (No Drop Action, Focus on Clearing Lines) ===
    # ==============================================================================
    # Only proceed with Phase 1 if the auto-drop check passed OR wasn't needed
    if autodrop_check_passed:
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_1_NAME} {'='*30}")

            # --- Environment Setup (Phase 1) ---
            write_log(f"Creating Environment for {PHASE_1_NAME}...")
            # Use DummyVecEnv which internally uses Monitor
            # Pass the configuration directly to the env creation function
            train_env_p1_base = DummyVecEnv([make_tetris_env(env_config=config_p1, seed=0)])
            # Stack frames after Monitor and DummyVecEnv
            train_env_p1_stacked = VecFrameStack(train_env_p1_base, n_stack=config_p1["n_stack"], channels_order="first")
            # Normalize rewards, not observations (images)
            train_env_p1 = VecNormalize(train_env_p1_stacked, norm_obs=False, norm_reward=True, gamma=config_p1["gamma"], clip_reward=10.0)

            action_space_size_p1 = train_env_p1.action_space.n
            write_log(f" Phase 1 Env Created. Action Space Size: {action_space_size_p1}")
            # Sanity check action space vs config
            expected_actions_p1 = 4 if config_p1["remove_drop_action"] else 5
            if action_space_size_p1 != expected_actions_p1:
                 write_log(f"⚠️ WARNING: Phase 1 Config 'remove_drop_action' is {config_p1['remove_drop_action']} (expected {expected_actions_p1} actions), but created env has {action_space_size_p1} actions!")


            # --- Callbacks (Phase 1) ---
            write_log("Setting up Callbacks for Phase 1...")
            callback_list_p1 = []
            # Curriculum Callback (inactive in P1 as start/end penalties are 0)
            curriculum_cb_p1 = CurriculumCallback(
                penalty_start=config_p1["penalty_game_over_start_coeff"],
                penalty_end=config_p1["penalty_game_over_end_coeff"],
                anneal_fraction=config_p1["curriculum_anneal_fraction"],
                total_training_steps=config_p1["total_timesteps"],
                verbose=1)
            callback_list_p1.append(curriculum_cb_p1)
            # Wandb Callback (if enabled)
            if wandb_enabled and run:
                wandb_cb_p1 = WandbCallback(
                     # No intermediate model saving via callback needed if saving at end of phase
                     model_save_path=None,
                     log="all", # Log default SB3 metrics
                     verbose=0)
                callback_list_p1.append(wandb_cb_p1)
                write_log(" Phase 1 Callbacks: Curriculum(Inactive), Wandb.")
            else:
                write_log(" Phase 1 Callbacks: Curriculum(Inactive).")

            # --- PPO Model (Phase 1 - Create New) ---
            write_log(f"Setting up NEW PPO Model for {PHASE_1_NAME}...")
            # Define TensorBoard log path (used by SB3 PPO and captured by Wandb if sync_tensorboard=True)
            tb_log_path_p1 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_1_NAME) if wandb_enabled else None

            model_p1 = PPO(
                policy=config_p1["policy_type"],
                env=train_env_p1,
                verbose=1, # Log training progress
                gamma=config_p1["gamma"],
                learning_rate=float(config_p1["learning_rate"]), # Ensure float
                n_steps=config_p1["n_steps"],
                batch_size=config_p1["batch_size"],
                n_epochs=config_p1["n_epochs"],
                gae_lambda=config_p1["gae_lambda"],
                clip_range=config_p1["clip_range"], # Can be a float or a schedule function
                ent_coef=config_p1["ent_coef"],
                vf_coef=0.5,         # Default value, can be tuned
                max_grad_norm=0.5,   # Default value, can be tuned
                seed=42,             # For reproducibility
                device=device,       # Use detected device ('cuda' or 'cpu')
                tensorboard_log=tb_log_path_p1, # Log to phase-specific directory
                policy_kwargs=dict(normalize_images=True) # Normalize images within the policy network
            )
            write_log(f" PPO Model created. Policy: {config_p1['policy_type']}, Device: {model_p1.device}")
            write_log(f" TensorBoard Log Path (P1): {tb_log_path_p1}")

            # --- Training (Phase 1) ---
            write_log(f"🚀 Starting Phase 1 Training ({config_p1['total_timesteps']:,} steps)...")
            t_start_p1 = time.time()
            model_p1.learn(
                total_timesteps=config_p1["total_timesteps"],
                callback=callback_list_p1,
                log_interval=10, # Log metrics every 10 updates (default is 100 for PPO)
                tb_log_name=PHASE_1_NAME, # Name for the TensorBoard run logs
                reset_num_timesteps=True # Start timestep counter at 0 for this phase
            )
            t_end_p1 = time.time()
            write_log(f"✅ Phase 1 Training Complete! Time: {(t_end_p1 - t_start_p1) / 3600:.2f} hours")

            # --- Saving (Phase 1 - Intermediate) ---
            write_log(f"💾 Saving Phase 1 model to: {phase1_model_save_path}")
            model_p1.save(phase1_model_save_path)
            write_log(f"💾 Saving Phase 1 VecNormalize stats to: {phase1_stats_save_path}")
            train_env_p1.save(phase1_stats_save_path)
            phase1_success = True # Mark Phase 1 as successful

        except Exception as e:
            write_log(f"❌❌❌ PHASE 1 ERROR: {e}", True)
            phase1_success = False # Mark as failed
        except KeyboardInterrupt:
            write_log(f"🛑 Phase 1 Training Interrupted by User.")
            phase1_success = False # Mark as unsuccessful if interrupted
        finally:
            # Ensure Phase 1 environment resources are closed
            if train_env_p1 is not None:
                try:
                    train_env_p1.close()
                    write_log(" Phase 1 Environment closed.")
                except Exception as e:
                    write_log(f" Error closing Phase 1 Env: {e}")

    # else: # Case where autodrop_check_passed is False is handled by phase1_success remaining False
        # write_log(f"⏩ Phase 1 was skipped due to failed auto-drop check.")


    # ==============================================================================
    # === PHASE 2: Continue Training (Add Drop Action, Add GO Penalty Curriculum) ===
    # ==============================================================================
    # Check if Phase 1 succeeded AND the necessary files were saved
    if phase1_success and os.path.exists(phase1_model_save_path) and os.path.exists(phase1_stats_save_path):
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_2_NAME} {'='*30}")

            # --- Environment Setup (Phase 2) ---
            write_log(f"Creating Environment for {PHASE_2_NAME}...")
            # Use DummyVecEnv which internally uses Monitor
            train_env_p2_base = DummyVecEnv([make_tetris_env(env_config=config_p2, seed=1)]) # Use diff seed
            # Stack frames
            train_env_p2_stacked = VecFrameStack(train_env_p2_base, n_stack=config_p2["n_stack"], channels_order="first")
            # Load P1 VecNormalize stats into the P2 environment wrapper
            write_log(f"🔄 Loading Phase 1 VecNormalize stats from: {phase1_stats_save_path}")
            train_env_p2 = VecNormalize.load(phase1_stats_save_path, train_env_p2_stacked)
            # CRITICAL: Set the environment to training mode after loading stats
            train_env_p2.training = True
            # Update reward normalization parameters if needed (optional)
            # train_env_p2.gamma = config_p2["gamma"] # Gamma should ideally match model if changed

            action_space_size_p2 = train_env_p2.action_space.n
            write_log(f" Phase 2 Env Created and Stats Loaded. Action Space Size: {action_space_size_p2}")
            # Sanity check action space
            if action_space_size_p2 != 5:
                 write_log(f"⚠️ WARNING: Phase 2 expected 5 actions (remove_drop_action=False), but created env has {action_space_size_p2} actions!")


            # --- Callbacks (Phase 2) ---
            write_log("Setting up Callbacks for Phase 2...")
            callback_list_p2 = []
            # Curriculum Callback (ACTIVE in P2 for GO Penalty)
            curriculum_cb_p2 = CurriculumCallback(
                penalty_start=config_p2["penalty_game_over_start_coeff"],
                penalty_end=config_p2["penalty_game_over_end_coeff"],
                anneal_fraction=config_p2["curriculum_anneal_fraction"],
                total_training_steps=config_p2["total_timesteps"],
                verbose=1)
            callback_list_p2.append(curriculum_cb_p2)
            # Wandb Callback
            if wandb_enabled and run:
                wandb_cb_p2 = WandbCallback(model_save_path=None, log="all", verbose=0)
                callback_list_p2.append(wandb_cb_p2)
                write_log(" Phase 2 Callbacks: Curriculum(Active), Wandb.")
            else:
                write_log(" Phase 2 Callbacks: Curriculum(Active).")

            # --- Load P1 Model & Adapt (Phase 2) ---
            write_log(f"🔄 Loading Phase 1 PPO model from: {phase1_model_save_path}")
            tb_log_path_p2 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_2_NAME) if wandb_enabled else None

            # Load the P1 model, specifying the *new* environment (P2 env)
            # This automatically adapts the model's action space if dimensions differ (PPO handles this)
            model_p2 = PPO.load(
                phase1_model_save_path,
                env=train_env_p2, # CRITICAL: Associate with the new env instance
                device=device,
                # custom_objects can sometimes be needed for complex changes, but PPO handles action space adaptation
                tensorboard_log=tb_log_path_p2 # Set new TensorBoard path for this phase
                # Optional: force_reset=False ensures LR schedule continues if one was used
            )
            write_log(" Phase 1 Model loaded. Policy action space potentially adapted (4 -> 5).")
            write_log(f" TensorBoard Log Path (P2): {tb_log_path_p2}")


            # Update model hyperparameters for Phase 2 (e.g., learning rate)
            write_log(f" Updating model hyperparameters for Phase 2 (LR={config_p2['learning_rate']:.1e}, EntCoef={config_p2['ent_coef']:.2f})...")
            model_p2.learning_rate = float(config_p2["learning_rate"])
            model_p2.ent_coef = float(config_p2["ent_coef"])
            # Clip range might need a schedule, handle appropriately if P1 used one.
            # If P1 used float, just setting float is fine. If schedule, need to create new schedule.
            # Assuming float for simplicity here:
            model_p2.clip_range = float(config_p2["clip_range"])
            # If using a learning rate schedule, you might need to reset it or adjust its progress.
            # Example: Reset LR schedule (if applicable)
            # if hasattr(model_p2.policy.optimizer.param_groups[0], 'initial_lr'):
            #     model_p2.policy.optimizer.param_groups[0]['lr'] = model_p2.learning_rate

            write_log(" Model hyperparameters updated.")

            # --- Training (Phase 2) ---
            write_log(f"🚀 Starting Phase 2 Training ({config_p2['total_timesteps']:,} steps)...")
            t_start_p2 = time.time()
            model_p2.learn(
                total_timesteps=config_p2["total_timesteps"],
                callback=callback_list_p2,
                log_interval=10,
                tb_log_name=PHASE_2_NAME,
                reset_num_timesteps=False # IMPORTANT: Continue timestep count from Phase 1 model
            )
            t_end_p2 = time.time()
            write_log(f"✅ Phase 2 Training Complete! Time: {(t_end_p2 - t_start_p2) / 3600:.2f} hours")

            # --- Saving (Phase 2 - Intermediate) ---
            write_log(f"💾 Saving Phase 2 model to: {phase2_model_save_path}")
            model_p2.save(phase2_model_save_path)
            write_log(f"💾 Saving Phase 2 VecNormalize stats to: {phase2_stats_save_path}")
            train_env_p2.save(phase2_stats_save_path)
            phase2_success = True # Mark Phase 2 as successful

        except Exception as e:
            write_log(f"❌❌❌ PHASE 2 ERROR: {e}", True)
            phase2_success = False
        except KeyboardInterrupt:
            write_log(f"🛑 Phase 2 Training Interrupted by User.")
            phase2_success = False
        finally:
            # Close Phase 2 Env
            if train_env_p2 is not None:
                try:
                    train_env_p2.close()
                    write_log(" Phase 2 Environment closed.")
                except Exception as e:
                    write_log(f" Error closing Phase 2 Env: {e}")
    else:
        # Log skipping Phase 2 only if Phase 1 was attempted but failed/skipped
        if not phase1_success:
             write_log(f"\n⏩ Skipping Phase 2 ({PHASE_2_NAME}) because Phase 1 did not complete successfully or its files are missing.")


    # ==============================================================================
    # === PHASE 3: Final Tuning (Add Height/Hole Penalties) ===
    # ==============================================================================
     # Check if Phase 2 succeeded AND the necessary files were saved
    if phase2_success and os.path.exists(phase2_model_save_path) and os.path.exists(phase2_stats_save_path):
        try:
            write_log(f"\n{'='*30} STARTING {PHASE_3_NAME} {'='*30}")

            # --- Environment Setup (Phase 3) ---
            write_log(f"Creating Environment for {PHASE_3_NAME}...")
            # Use DummyVecEnv which internally uses Monitor
            train_env_p3_base = DummyVecEnv([make_tetris_env(env_config=config_p3, seed=2)]) # Use diff seed
            # Stack frames
            train_env_p3_stacked = VecFrameStack(train_env_p3_base, n_stack=config_p3["n_stack"], channels_order="first")
            # Load P2 VecNormalize stats into P3 Env Wrapper
            write_log(f"🔄 Loading Phase 2 VecNormalize stats from: {phase2_stats_save_path}")
            train_env_p3 = VecNormalize.load(phase2_stats_save_path, train_env_p3_stacked)
            train_env_p3.training = True # Set to training mode
            # train_env_p3.gamma = config_p3["gamma"] # Update gamma if changed

            action_space_size_p3 = train_env_p3.action_space.n
            write_log(f" Phase 3 Env Created and Stats Loaded. Action Space Size: {action_space_size_p3}")
            if action_space_size_p3 != 5:
                 write_log(f"⚠️ WARNING: Phase 3 expected 5 actions, but created env has {action_space_size_p3} actions!")

            # --- Callbacks (Phase 3) ---
            write_log("Setting up Callbacks for Phase 3...")
            callback_list_p3 = []
            # Curriculum Callback (Inactive in P3 as GO penalty is fixed)
            curriculum_cb_p3 = CurriculumCallback(
                penalty_start=config_p3["penalty_game_over_start_coeff"], # Start = End
                penalty_end=config_p3["penalty_game_over_end_coeff"],
                anneal_fraction=config_p3["curriculum_anneal_fraction"], # Should be 0
                total_training_steps=config_p3["total_timesteps"],
                verbose=1)
            callback_list_p3.append(curriculum_cb_p3)
            # Wandb Callback
            if wandb_enabled and run:
                wandb_cb_p3 = WandbCallback(model_save_path=None, log="all", verbose=0)
                callback_list_p3.append(wandb_cb_p3)
                write_log(" Phase 3 Callbacks: Curriculum(Inactive), Wandb.")
            else:
                write_log(" Phase 3 Callbacks: Curriculum(Inactive).")

            # --- Load P2 Model & Adapt (Phase 3) ---
            write_log(f"🔄 Loading Phase 2 PPO model from: {phase2_model_save_path}")
            tb_log_path_p3 = os.path.join(output_dir, "runs", TOTAL_RUNTIME_ID, PHASE_3_NAME) if wandb_enabled else None
            # Load P2 model, associate with P3 environment
            model_p3 = PPO.load(
                phase2_model_save_path,
                env=train_env_p3, # Use P3 env
                device=device,
                tensorboard_log=tb_log_path_p3
            )
            write_log(" Phase 2 Model loaded.")
            write_log(f" TensorBoard Log Path (P3): {tb_log_path_p3}")

            # Update model hyperparameters for Phase 3
            write_log(f" Updating model hyperparameters for Phase 3 (LR={config_p3['learning_rate']:.1e}, EntCoef={config_p3['ent_coef']:.2f})...")
            model_p3.learning_rate = float(config_p3["learning_rate"])
            model_p3.ent_coef = float(config_p3["ent_coef"])
            model_p3.clip_range = float(config_p3["clip_range"]) # Assuming float
            # Optional: Reset LR schedule if needed

            write_log(" Model hyperparameters updated.")

            # --- Training (Phase 3) ---
            write_log(f"🚀 Starting Phase 3 Training ({config_p3['total_timesteps']:,} steps)...")
            t_start_p3 = time.time()
            model_p3.learn(
                total_timesteps=config_p3["total_timesteps"],
                callback=callback_list_p3,
                log_interval=10,
                tb_log_name=PHASE_3_NAME,
                reset_num_timesteps=False # Continue timestep count
            )
            t_end_p3 = time.time()
            write_log(f"✅ Phase 3 Training Complete! Time: {(t_end_p3 - t_start_p3) / 3600:.2f} hours")

            # --- Saving (Phase 3 - FINAL) ---
            write_log(f"💾 Saving FINAL Phase 3 model to: {phase3_final_model_save_path}")
            model_p3.save(phase3_final_model_save_path)
            write_log(f"💾 Saving FINAL Phase 3 VecNormalize stats to: {phase3_final_stats_save_path}")
            train_env_p3.save(phase3_final_stats_save_path)

            # Display final file links if in IPython
            if ipython_available:
                 write_log("Displaying final model/stats file links:")
                 display(FileLink(phase3_final_model_save_path))
                 display(FileLink(phase3_final_stats_save_path))

            phase3_success = True # Mark Phase 3 as successful

        except Exception as e:
            write_log(f"❌❌❌ PHASE 3 ERROR: {e}", True)
            phase3_success = False
        except KeyboardInterrupt:
            write_log(f"🛑 Phase 3 Training Interrupted by User.")
            phase3_success = False
        finally:
            # Close P3 Env
            if train_env_p3 is not None:
                try:
                    train_env_p3.close()
                    write_log(" Phase 3 Environment closed.")
                except Exception as e:
                    write_log(f" Error closing Phase 3 Env: {e}")
    else:
         # Log skipping Phase 3 only if Phase 2 was attempted but failed
         if not phase2_success and phase1_success: # Check P1 success to avoid double logging if P1 also failed
              write_log(f"\n⏩ Skipping Phase 3 ({PHASE_3_NAME}) because Phase 2 did not complete successfully or its files are missing.")


except Exception as main_e:
    # Catch any unhandled exceptions in the main script execution
    write_log(f"💥💥💥 UNHANDLED EXCEPTION in main script execution: {main_e}", True)
except KeyboardInterrupt:
     # Handle user interruption gracefully
     write_log("\n🛑🛑🛑 Main script execution interrupted by user (Ctrl+C). 🛑🛑🛑")
finally:
    # ==============================================================================
    # === Final Cleanup & Reporting ===
    # ==============================================================================
    write_log(f"\n{'='*20} Final Cleanup & Reporting {'='*20}")

    # --- Terminate Java Server ---
    if java_process and java_process.poll() is None: # Check if process exists and is running
        write_log("🧹 Terminating Java server process...")
        java_process.terminate() # Ask nicely first
        try:
            java_process.wait(timeout=5) # Wait for termination
            write_log("✅ Java server terminated gracefully.")
        except subprocess.TimeoutExpired:
            write_log("⚠️ Java server did not terminate gracefully, killing...")
            java_process.kill() # Force kill
            try:
                 java_process.wait(timeout=2) # Wait briefly for kill
                 write_log("✅ Java server killed.")
            except Exception as kill_e:
                 write_log(f"⚠️ Error waiting for killed Java process: {kill_e}")
        except Exception as e:
           write_log(f"⚠️ Error during Java server termination: {e}")
    elif java_process:
        write_log("🧹 Java server process already terminated.")
    else:
        write_log("🧹 Java server process was not started or failed early.")

    # --- Upload Final Successful Artifacts to Wandb ---
    final_model_to_upload = None
    final_stats_to_upload = None
    final_phase_name = "None"

    # Determine the latest successfully completed phase's artifacts
    if phase3_success and os.path.exists(phase3_final_model_save_path) and os.path.exists(phase3_final_stats_save_path):
        final_model_to_upload = phase3_final_model_save_path
        final_stats_to_upload = phase3_final_stats_save_path
        final_phase_name = PHASE_3_NAME
    elif phase2_success and os.path.exists(phase2_model_save_path) and os.path.exists(phase2_stats_save_path):
        final_model_to_upload = phase2_model_save_path # Use P2 temp files if P3 failed
        final_stats_to_upload = phase2_stats_save_path
        final_phase_name = PHASE_2_NAME
    elif phase1_success and os.path.exists(phase1_model_save_path) and os.path.exists(phase1_stats_save_path):
        final_model_to_upload = phase1_model_save_path # Use P1 temp files if P2/P3 failed
        final_stats_to_upload = phase1_stats_save_path
        final_phase_name = PHASE_1_NAME

    if wandb_enabled and run and final_model_to_upload:
        write_log(f"☁️ Uploading final successful artifacts (from {final_phase_name}) to Wandb...")
        try:
            # Check if the run is still active before saving
            if wandb.run and wandb.run.id == run.id:
                 # Save model as artifact
                 model_artifact = wandb.Artifact(f"{STUDENT_ID}-ppo-model-{TOTAL_RUNTIME_ID}", type="model")
                 model_artifact.add_file(final_model_to_upload)
                 run.log_artifact(model_artifact)
                 write_log(f"  Model artifact logged: {os.path.basename(final_model_to_upload)}")

                 # Save stats as artifact
                 stats_artifact = wandb.Artifact(f"{STUDENT_ID}-vecnorm-stats-{TOTAL_RUNTIME_ID}", type="normalization_stats")
                 stats_artifact.add_file(final_stats_to_upload)
                 run.log_artifact(stats_artifact)
                 write_log(f"  Stats artifact logged: {os.path.basename(final_stats_to_upload)}")

                 write_log("✅ Wandb artifact upload successful.")
            else:
                 write_log("⚠️ Wandb run appears inactive, cannot upload final artifacts.")
        except Exception as e:
            write_log(f"⚠️ Wandb artifact upload error: {e}", True)
    elif wandb_enabled and run:
        write_log("☁️ Skipping final artifact upload as no phase completed successfully or final files are missing.")

    # --- Finish Wandb Run ---
    if wandb_enabled and run:
        write_log("Finishing Wandb run...")
        # Determine overall success based on completing all intended phases (or latest successful)
        overall_success = phase3_success # Define success as completing Phase 3
        exit_code = 0 if overall_success else 1 # 0 for success, 1 for failure
        try:
            # Ensure run is still active before finishing
            if wandb.run and wandb.run.id == run.id:
                # Add summary metrics if desired
                run.summary["Phase1_Success"] = phase1_success
                run.summary["Phase2_Success"] = phase2_success
                run.summary["Phase3_Success"] = phase3_success
                run.summary["Overall_Success"] = overall_success
                run.finish(exit_code=exit_code) # Finish with exit code
                write_log(f"✅ Wandb run '{TOTAL_RUNTIME_ID}' finished with exit code {exit_code}.")
            else:
                write_log("⚠️ Wandb run was already finished or inactive.")
        except Exception as finish_e:
            write_log(f"⚠️ Error finishing Wandb run: {finish_e}")

    # --- Final Status Report ---
    write_log(f"\n🏁🏁🏁 TRAINING RUN SUMMARY ({TOTAL_RUNTIME_ID}) 🏁🏁🏁")
    write_log(f"  Phase 1 ({PHASE_1_NAME}) Success: {phase1_success}")
    write_log(f"  Phase 2 ({PHASE_2_NAME}) Success: {phase2_success}")
    write_log(f"  Phase 3 ({PHASE_3_NAME}) Success: {phase3_success}")
    final_result_path = "N/A"
    if phase3_success: final_result_path = phase3_final_model_save_path
    elif phase2_success: final_result_path = phase2_model_save_path
    elif phase1_success: final_result_path = phase1_model_save_path
    write_log(f"  Final Model Available: {final_result_path}")


    # --- Display Log File Link ---
    write_log("-" * 50)
    if os.path.exists(log_path):
        write_log(f"📜 Training log saved to: {log_path}")
        if ipython_available:
            try:
                display(FileLink(log_path))
            except Exception as display_e:
                 write_log(f"(Could not display log file link: {display_e})")
    else:
        write_log("📜 Log file not found.")

    write_log("🏁 Script execution finished. 🏁")