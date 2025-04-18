# -*- coding: utf-8 -*-
# <<< ALL IMPORTS FROM YOUR ORIGINAL CODE + LOGGER >>>
import numpy as np
# import wandb # wandb import moved lower
from wandb import Settings
import socket
import cv2
import subprocess
import os
import shutil
import glob
import imageio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as configure_logger # For resetting logger
from IPython.display import FileLink, display
import torch
import time
import pygame
import traceback

# --- Wandb Setup --- (Same as your provided code)
try:
    from kaggle_secrets import UserSecretsClient
    secrets_available = True
except ImportError:
    secrets_available = False
    print("Kaggle secrets not available.")

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb_module_available = True
except ImportError:
    wandb_module_available = False
    print("Wandb module not found.")
    # Dummy callback with proper __init__
    class WandbCallback(BaseCallback):
        def __init__(self, *args, **kwargs):
            allowed_kwargs = {k: v for k, v in kwargs.items() if k in ['verbose']}
            super().__init__(**allowed_kwargs)
        def _on_step(self)->bool:
            return True

# --- Configuration ---
STUDENT_ID = "113598065" # From your original code
TOTAL_RUNTIME_ID = f"ppo_3phase_{int(time.time())}" # Overall Run ID

# ========================= PHASE 1 CONFIG =========================
PHASE_1_NAME = "Phase1_ClearLines_NoDrop"
PHASE_1_TIMESTEPS = 3000000 # Longer Phase 1
config_p1 = {
    "phase_name": PHASE_1_NAME, "total_timesteps": PHASE_1_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_1_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.15, "learning_rate": 1e-4, "n_stack": 4, "student_id": STUDENT_ID,
    # Phase 1 Rewards: Focus on line clear (exponential), minimal step, no penalties
    "reward_line_clear_coeff": 100.0, "penalty_height_increase_coeff": 0.0, "penalty_hole_increase_coeff": 0.0,
    "penalty_step_coeff": 0.05, "penalty_game_over_start_coeff": 0.0, "penalty_game_over_end_coeff": 0.0,
    "curriculum_anneal_fraction": 0.0, "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}, "remove_drop_action": True
}

# ========================= PHASE 2 CONFIG =========================
PHASE_2_NAME = "Phase2_AddDrop_AddGO"
PHASE_2_TIMESTEPS = 2000000 # Phase 2 duration
config_p2 = {
    "phase_name": PHASE_2_NAME, "total_timesteps": PHASE_2_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_2_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.1, "learning_rate": 5e-5, "n_stack": 4, "student_id": STUDENT_ID,
    # Phase 2 Rewards: Add GO penalty curriculum, keep other penalties 0
    "reward_line_clear_coeff": 100.0, "penalty_height_increase_coeff": 0.0, "penalty_hole_increase_coeff": 0.0, # Still 0
    "penalty_step_coeff": 0.05, "penalty_game_over_start_coeff": 0.0, "penalty_game_over_end_coeff": 20.0, # Add GO pen
    "curriculum_anneal_fraction": 0.5, "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0}, "remove_drop_action": False # Add drop
}

# ========================= PHASE 3 CONFIG =========================
PHASE_3_NAME = "Phase3_AddStackPenalty"
PHASE_3_TIMESTEPS = 1500000 # Phase 3 duration
config_p3 = {
    "phase_name": PHASE_3_NAME, "total_timesteps": PHASE_3_TIMESTEPS, "env_id": f"TetrisEnv-v1-{PHASE_3_NAME}",
    "policy_type": "CnnPolicy", "n_steps": 1024, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.1, "ent_coef": 0.05, "learning_rate": 1e-5, "n_stack": 4, "student_id": STUDENT_ID,
    # Phase 3 Rewards: Add Height/Hole penalties, keep GO penalty fixed
    "reward_line_clear_coeff": 100.0,
    "penalty_height_increase_coeff": 0.5, # <<< Add Height Penalty >>>
    "penalty_hole_increase_coeff": 1.0,   # <<< Add Hole Penalty >>>
    "penalty_step_coeff": 0.05,
    "penalty_game_over_start_coeff": config_p2["penalty_game_over_end_coeff"], # Start at Phase 2 end value
    "penalty_game_over_end_coeff": config_p2["penalty_game_over_end_coeff"],   # Keep it constant
    "curriculum_anneal_fraction": 0.0, # No annealing
    "line_clear_multipliers": {1: 1.0, 2: 3.0, 3: 5.0, 4: 8.0},
    "remove_drop_action": False # Keep drop action
}

# --- Wandb Login --- (Using the logic from your original code)
wandb_enabled = False
WANDB_API_KEY = None
if wandb_module_available:
    try:
        if secrets_available:
            print("Attempting to load WANDB_API_KEY from Kaggle Secrets...")
            user_secrets = UserSecretsClient()
            WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
            print("Secret loaded from Kaggle.")
        elif "WANDB_API_KEY" in os.environ:
            print("Attempting to load WANDB_API_KEY from environment variable...")
            WANDB_API_KEY = os.environ["WANDB_API_KEY"]
            print("Secret loaded from environment variable.")

        if WANDB_API_KEY:
            print("Attempting to login to Wandb...")
            try:
                wandb.login(key=WANDB_API_KEY, timeout=30)
                wandb_enabled = True
                print("✅ Wandb login successful.")
            except Exception as login_e:
                print(f"Wandb login attempt failed: {login_e}. Running without Wandb logging.")
                wandb_enabled = False
                WANDB_API_KEY = None # Clear key on failure
        else:
            print("WANDB_API_KEY not found. Running without Wandb logging.")

    except Exception as e:
        print(f"Wandb setup failed during secret retrieval: {e}. Running without Wandb logging.")
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
            project=project_name, id=TOTAL_RUNTIME_ID, name=f"Run_{TOTAL_RUNTIME_ID}",
            # entity="t113598065-ntut-edu-tw", # Your entity if needed
            sync_tensorboard=True, monitor_gym=True, save_code=True,
            settings=Settings(init_timeout=180),
            config={ "Phase1": config_p1, "Phase2": config_p2, "Phase3": config_p3 }, # Log all configs
            resume="allow"
        )
        print(f"✅ Wandb run initialized. Run ID: {run.id if run else 'N/A'}")
    except Exception as e:
        print(f"Wandb init failed: {e}.")
        wandb_enabled = False
        run = None

# --- Log & File Paths ---
log_path = f"/kaggle/working/tetris_train_log_{TOTAL_RUNTIME_ID}.txt"
# Temporary paths for intermediate results
phase1_model_save_path = f"/kaggle/working/{STUDENT_ID}_ppo_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.zip"
phase1_stats_save_path = f"/kaggle/working/vecnormalize_stats_{PHASE_1_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl"
phase2_model_save_path = f"/kaggle/working/{STUDENT_ID}_ppo_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.zip"
phase2_stats_save_path = f"/kaggle/working/vecnormalize_stats_{PHASE_2_NAME}_temp_{TOTAL_RUNTIME_ID}.pkl"
# Final paths for Phase 3 results
phase3_final_model_save_path = f"/kaggle/working/{STUDENT_ID}_ppo_{PHASE_3_NAME}_final_{TOTAL_RUNTIME_ID}.zip"
phase3_final_stats_save_path = f"/kaggle/working/vecnormalize_stats_{PHASE_3_NAME}_final_{TOTAL_RUNTIME_ID}.pkl"

# --- Helper Functions & Server Start --- (Using helpers from your original code)
def write_log(message, exc_info=False): # Your write_log function
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    print(log_message)
    if exc_info:
        log_message += "\n" + traceback.format_exc()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Use 'with' statement for safer file handling
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Log write error: {e}")

def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60): # Your wait function
    write_log(f"⏳ Waiting Tetris server @ {ip}:{port}...")
    t_start = time.time()
    while True:
        try:
            # Use with statement for socket
            with socket.create_connection((ip, port), timeout=1.0) as s:
                pass # Connection successful
            write_log("✅ Java server ready.")
            return True
        except socket.error:
            if time.time() - t_start > timeout:
                write_log(f"❌ Server timeout ({timeout}s)")
                return False
            time.sleep(1.0)

java_process = None
try: # Your Java server start logic
    write_log("🚀 Starting Java Tetris server...")
    jar_file = "TetrisTCPserver_v0.6.jar"
    if not os.path.exists(jar_file):
        raise FileNotFoundError(f"JAR file not found: '{jar_file}' at '{os.getcwd()}'")
    # Use Popen arguments correctly
    java_process = subprocess.Popen(
        ["java", "-jar", jar_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    write_log(f"✅ Java server process started (PID: {java_process.pid})")
    if not wait_for_tetris_server():
        raise TimeoutError("Java server did not become available.")
except Exception as e:
    write_log(f"❌ Java server error: {e}", True)
    # Ensure process termination if startup fails
    if java_process and java_process.poll() is None:
        java_process.terminate()
        try: java_process.wait(timeout=1)
        except subprocess.TimeoutExpired: java_process.kill()
    raise

# Check GPU availability
if torch.cuda.is_available():
    write_log(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    write_log("⚠️ Using CPU.")

# --------------------------------------------------------------------------
# <<< TetrisEnv Class Definition >>>
# Based on your original structure, modified for phased config & exp rewards
# !! Formatting fixed !!
# --------------------------------------------------------------------------
class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS_NO_DROP = 4
    N_DISCRETE_ACTIONS_WITH_DROP = 5
    IMG_HEIGHT = 200
    IMG_WIDTH = 100
    IMG_CHANNELS = 3
    RESIZED_DIM = 84

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None, env_config=None):
        super().__init__()
        self.render_mode = render_mode
        current_config = env_config
        if current_config is None:
            raise ValueError("env_config must be provided to TetrisEnv")

        self.config_remove_drop = current_config.get("remove_drop_action", False)

        # Set action space based on config
        if self.config_remove_drop:
            self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS_NO_DROP)
            self.command_map = {0: b"move -1\n", 1: b"move 1\n", 2: b"rotate 0\n", 3: b"rotate 1\n"}
            self._log_prefix = "[Env NoDrop]"
        else:
            self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS_WITH_DROP)
            self.command_map = {0: b"move -1\n", 1: b"move 1\n", 2: b"rotate 0\n", 3: b"rotate 1\n", 4: b"drop\n"}
            self._log_prefix = "[Env Drop]"

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), dtype=np.uint8)
        self.server_ip = host_ip
        self.server_port = host_port
        self.client_sock = None
        self._connect_socket() # Initial connection attempt

        # Internal state variables
        self.current_cumulative_lines = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None

        # Load reward params from CURRENT phase config
        self.reward_line_clear_coeff = current_config["reward_line_clear_coeff"]
        self.penalty_height_increase_coeff = current_config["penalty_height_increase_coeff"]
        self.penalty_hole_increase_coeff = current_config["penalty_hole_increase_coeff"]
        self.penalty_step_coeff = current_config["penalty_step_coeff"]
        self.line_clear_multipliers = current_config["line_clear_multipliers"]
        self.penalty_game_over_start_coeff = current_config["penalty_game_over_start_coeff"]
        self.penalty_game_over_end_coeff = current_config["penalty_game_over_end_coeff"]
        self.current_go_penalty = self.penalty_game_over_start_coeff # Initialize with start value
        self.current_phase_name = current_config.get('phase_name', 'UnknownPhase')

        # Log initialization details clearly
        write_log(f"{self._log_prefix} Initialized Phase: {self.current_phase_name}")
        write_log(f"{self._log_prefix} Action Space Size: {self.action_space.n}")
        write_log(f"{self._log_prefix} Initial Rewards: LC_Base={self.reward_line_clear_coeff:.1f}, Step={self.penalty_step_coeff:.2f}, GO_Start={self.current_go_penalty:.1f}")
        write_log(f"{self._log_prefix} Initial Penalties: H={self.penalty_height_increase_coeff:.1f}, O={self.penalty_hole_increase_coeff:.1f}")

        # Pygame related
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False
        self._wandb_log_error_reported = False

    def set_game_over_penalty(self, new_penalty_value):
        """Allows the game over penalty coefficient to be updated externally by CurriculumCallback."""
        self.current_go_penalty = new_penalty_value

    def _initialize_pygame(self): # Your original method, properly formatted
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                import pygame
                pygame.init()
                pygame.display.init()
                self.window_surface = pygame.display.set_mode((self.RESIZED_DIM * 4, self.RESIZED_DIM * 4))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log("   Pygame initialized.")
            except Exception as e:
                write_log(f"⚠️ Pygame error: {e}")
                self.render_mode = None # Disable human rendering on error

    def _connect_socket(self): # Your original method, properly formatted
        try:
            # Close existing socket if any
            if self.client_sock:
                try:
                    self.client_sock.close()
                except socket.error:
                    pass # Ignore errors on close
            # Create and connect new socket
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0) # Connection timeout
            self.client_sock.connect((self.server_ip, self.server_port))
        except socket.error as e:
            raise ConnectionError(f"{self._log_prefix} Connect fail: {e}")

    def _send_command(self, command: bytes): # Your original method, properly formatted
        if not self.client_sock:
            raise ConnectionError(f"{self._log_prefix} Socket is not connected for send.")
        try:
            self.client_sock.sendall(command)
        except socket.timeout:
            raise ConnectionAbortedError(f"{self._log_prefix} Send command timeout.")
        except socket.error as e:
            raise ConnectionAbortedError(f"{self._log_prefix} Send command error: {e}")

    def _receive_data(self, size: int): # Your original method, properly formatted
        if not self.client_sock:
            raise ConnectionError(f"{self._log_prefix} Socket is not connected for receive.")
        data = b""
        self.client_sock.settimeout(10.0) # Timeout for blocking recv calls
        t_start = time.time()
        while len(data) < size:
            if time.time() - t_start > 10.0: # Overall timeout for receiving 'size' bytes
                raise socket.timeout(f"Timeout receiving {size} bytes (received {len(data)})")
            try:
                # Receive remaining bytes needed
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    # Socket closed unexpectedly by the server
                    raise ConnectionAbortedError(f"{self._log_prefix} Socket connection broken (received empty chunk).")
                data += chunk
            except socket.timeout:
                # Individual recv call timed out, continue loop if overall timeout not exceeded
                continue
            except socket.error as e:
                raise ConnectionAbortedError(f"{self._log_prefix} Receive data error: {e}")
        return data # Return the complete data

    def get_tetris_server_response(self): # Your original method, properly formatted
        try:
            term = (self._receive_data(1) == b'\x01')
            lines = int.from_bytes(self._receive_data(4), 'big')
            h = int.from_bytes(self._receive_data(4), 'big')
            holes = int.from_bytes(self._receive_data(4), 'big')
            sz = int.from_bytes(self._receive_data(4), 'big')

            if not 0 < sz <= 1000000: # Check image size validity
                raise ValueError(f"Invalid image size received: {sz}")

            img_data = self._receive_data(sz)
            nparr = np.frombuffer(img_data, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if np_image is None:
                write_log(f"{self._log_prefix} ❌ Image decode failed. Using last valid observation.")
                # Return game over true to end episode safely
                return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

            # Resize and convert to grayscale for observation
            res = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            self.last_raw_render_frame = res.copy() # Keep color frame for rendering
            g = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            obs = np.expand_dims(g, axis=0).astype(np.uint8)
            self.last_observation = obs.copy() # Store latest valid observation

            return term, lines, h, holes, obs

        except (ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError, ConnectionError, ValueError, socket.timeout) as e:
            write_log(f"{self._log_prefix} ❌ Network/Value/Timeout error getting response: {e}.")
            # Return game over true to end episode safely
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"{self._log_prefix} ❌ Unexpected error getting server response: {e}", True)
            # Return game over true to end episode safely
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action): # Uses exponential reward logic now, properly formatted
        act_val = action.item() if isinstance(action, np.ndarray) else action
        command = self.command_map.get(act_val)
        if command is None:
            # Handle invalid action if it somehow occurs
            command = self.command_map.get(2, b"rotate 0\n") # Default to NOP (e.g., rotate 0)
            write_log(f"⚠️ Invalid action received in step: {act_val}. Sending NOP.")

        try:
            self._send_command(command)
            term, nl, nh, no, obs = self.get_tetris_server_response()
        except (ConnectionAbortedError, ConnectionError, ValueError, socket.timeout) as e:
            # Handle communication errors during step
            write_log(f"{self._log_prefix} ❌ Step comm/value error: {e}. Ending episode.")
            reward = -self.current_go_penalty # Apply current GO penalty
            info = {'lines': self.current_cumulative_lines, 'l': self.lifetime, 'status': 'error', 'final_status': 'comm_error'}
            # Return state consistent with termination
            return self.last_observation.copy(), reward, True, True, info # terminated=True, truncated=True

        # Calculate reward components based on state change
        lcs = max(0, nl - self.current_cumulative_lines) # Lines cleared this step
        reward = 0.0
        lcr = 0.0 # Line clear reward component
        mult = 0.0 # Multiplier used

        # Exponential Line Clear Reward
        if lcs > 0:
            mult = self.line_clear_multipliers.get(lcs, self.line_clear_multipliers.get(4, 8.0)) # Default to 4+ multiplier
            lcr = mult * self.reward_line_clear_coeff
            reward += lcr

        # Height Penalty
        hi = max(0, nh - self.current_height) # Height increase
        hp = hi * self.penalty_height_increase_coeff # Height penalty value
        reward -= hp

        # Hole Penalty
        oi = max(0, no - self.current_holes) # Hole increase
        op = oi * self.penalty_hole_increase_coeff # Hole penalty value
        reward -= op

        # Step Reward (Survival Bonus)
        sr = self.penalty_step_coeff # Step reward value
        reward += sr

        # Game Over Penalty
        gop = 0.0 # Game over penalty value
        if term:
            gop = self.current_go_penalty # Use potentially annealed value
            reward -= gop
            # Log detailed reward breakdown on game over
            write_log(f"{self._log_prefix} 💔 GameOver L={nl} T={self.lifetime + 1} | "
                      f"R: LC(x{mult:.1f})={lcr:.1f} H={-hp:.1f} O={-op:.1f} S={sr:.2f} GO={-gop:.1f} "
                      f"-> Tot={reward:.1f}")

        # Update internal state AFTER calculating rewards based on change
        self.current_cumulative_lines = nl
        self.current_height = nh
        self.current_holes = no
        self.lifetime += 1

        # Prepare info dict
        info = {'lines': self.current_cumulative_lines, 'l': self.lifetime}
        if term:
            # Add terminal observation and episode stats for SB3 Monitor/Logger
            info['terminal_observation'] = obs.copy()
            info['episode'] = {
                'r': reward, # Note: SB3 usually uses summed reward, this is last step's reward
                'l': self.lifetime,
                'lines': self.current_cumulative_lines, # Final lines for episode
                'go_pen': gop # Log the penalty applied
            }

        # Log step details for Wandb
        log_dict = {
            "rew": reward, "lcr": lcr, "hp": -hp, "op": -op, "sr": sr, "gop": -gop,
            "lcs": lcs, "hi": hi, "oi": oi, "h": nh, "o": no, "l": self.lifetime,
            "go_coeff": self.current_go_penalty
        }
        if wandb_enabled and run:
            self._safe_wandb_log(log_dict)

        # Render if needed
        if self.render_mode == "human":
            self.render()

        # Return standard gym tuple: observation, reward, terminated, truncated, info
        # Assuming no truncation based on time limit here
        return obs, reward, term, False, info

    def reset(self, seed=None, options=None): # Your original method, properly formatted
        super().reset(seed=seed)
        self._wandb_log_error_reported = False # Reset wandb error flag
        for attempt in range(3):
            try:
                self._connect_socket()
                self._send_command(b"start\n")
                term, sl, h, o, obs = self.get_tetris_server_response()

                # Check if server reset correctly
                if term or sl != 0:
                    write_log(f"{self._log_prefix} ⚠️ Invalid reset state (Term={term}, Lines={sl}) on attempt {attempt + 1}. Retrying...")
                    time.sleep(0.5 + attempt * 0.5)
                    continue # Retry connection

                # Reset internal state on successful connection and valid server state
                self.current_cumulative_lines = 0
                self.current_height = h
                self.current_holes = o
                self.lifetime = 0
                self.last_observation = obs.copy()
                self.last_raw_render_frame = None # Clear render frame
                info = {'start_height': h, 'start_holes': o}
                return obs, info # Return initial observation and info

            except (ConnectionAbortedError, ConnectionError, ConnectionRefusedError, socket.error, TimeoutError, ValueError) as e:
                write_log(f"{self._log_prefix} 🔌 Reset connection/value error attempt {attempt + 1}: {e}")
                # Clean up socket before retrying
                if self.client_sock:
                    try: self.client_sock.close()
                    except: pass
                    self.client_sock = None
                if attempt == 2: # If last attempt failed
                    raise RuntimeError(f"Failed to reset environment after multiple connection attempts: {e}")
                time.sleep(1.0 + attempt * 0.5)
                # continue - implicit due to loop

            except Exception as e:
                write_log(f"{self._log_prefix} ❌ Unexpected reset error attempt {attempt + 1}: {e}", True)
                if attempt < 2:
                    time.sleep(1.0 + attempt * 0.5)
                else: # Raise error on last attempt
                    raise RuntimeError(f"Failed reset due to unexpected error: {e}")

        # If loop finishes without returning (shouldn't happen with raises)
        raise RuntimeError("Failed to reset environment after retry loop.")

    def render(self): # Your original method, properly formatted
        self._initialize_pygame()
        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None:
                return # Cannot render if window not created
            frame = self.last_raw_render_frame
            if frame is not None and frame.size > 0:
                try:
                    # Convert BGR (cv2 default) to RGB (pygame default)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Create pygame surface of the same size as the frame
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    # Blit array onto surface (transpose needed: Pygame is (width, height), numpy is (height, width))
                    pygame.surfarray.blit_array(surf, np.transpose(rgb_frame, (1, 0, 2)))
                    # Scale surface to fit the display window
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    # Draw the scaled surface onto the window at position (0,0)
                    self.window_surface.blit(surf, (0, 0))
                    # Process internal pygame events
                    pygame.event.pump()
                    # Update the full display surface to the screen
                    pygame.display.flip()
                    # Maintain frame rate
                    self.clock.tick(self.metadata["render_fps"])
                except Exception as e:
                    write_log(f"⚠️ Pygame render error: {e}")
            else:
                # If no frame available, draw a black screen
                try:
                    self.window_surface.fill((0, 0, 0))
                    pygame.display.flip()
                except Exception as e:
                    write_log(f"⚠️ Pygame fill error: {e}")
        elif self.render_mode == "rgb_array":
            # Return the last captured raw frame (color BGR from cv2), converted to RGB
            if self.last_raw_render_frame is not None:
                return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
            else:
                # Return a black frame if no frame has been captured yet
                return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self): # Your original method, properly formatted
        if self.client_sock:
            try:
                self.client_sock.close()
                write_log("   Socket closed.")
            except socket.error:
                pass # Ignore errors on close
            self.client_sock = None
        if self.is_pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                write_log("   Pygame closed.")
            except Exception:
                pass # Ignore errors on quit
            self.is_pygame_initialized = False

    def _safe_wandb_log(self, data): # Uses correct phase name, properly formatted
        if wandb_enabled and run:
            try:
                # Add phase prefix to keys before logging
                prefixed_data = {f"{self.current_phase_name}/{k}": v for k, v in data.items()}
                wandb.log(prefixed_data, commit=False) # commit=False for SB3 logger integration
            except Exception as e:
                if not self._wandb_log_error_reported:
                    write_log(f"⚠️ Wandb log error: {e}")
                    self._wandb_log_error_reported = True # Prevent spamming

# --------------------------------------------------------------------------
# <<< CurriculumCallback Definition >>>
# Based on your original structure, properly formatted
# --------------------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    def __init__(self, penalty_start: float, penalty_end: float, anneal_fraction: float, total_training_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.penalty_start = penalty_start
        self.penalty_end = penalty_end
        self.anneal_fraction = anneal_fraction
        self.total_training_steps = total_training_steps
        self.anneal_timesteps = int(total_training_steps * anneal_fraction)
        # Flags to prevent repeated logging
        self._annealing_finished_logged = False
        self._callback_error_logged = False
        # Log initialization
        if self.anneal_timesteps > 0:
            write_log(f"CurricCallback: GO Pen {penalty_start:.1f} -> {penalty_end:.1f} over {self.anneal_timesteps} steps.")
        else:
            write_log(f"CurricCallback: GO Pen fixed @ {penalty_start:.1f}.")

    def _on_step(self) -> bool:
        current_penalty = self.penalty_start # Default to start penalty

        # Calculate current penalty only if annealing is active for this phase
        if self.penalty_start != self.penalty_end and self.anneal_timesteps > 0:
            if self.num_timesteps <= self.anneal_timesteps:
                # Calculate progress, ensuring it's within [0, 1]
                progress = max(0.0, min(1.0, self.num_timesteps / self.anneal_timesteps))
                current_penalty = self.penalty_start + progress * (self.penalty_end - self.penalty_start)
            else: # Annealing finished
                current_penalty = self.penalty_end
                # Log only once when annealing finishes
                if not self._annealing_finished_logged:
                    if self.verbose > 0:
                        write_log(f" Curriculum annealing finished. GO Penalty fixed at: {current_penalty:.1f}")
                    self._annealing_finished_logged = True # Set flag

        # Update penalty in the environment(s) via VecEnv method
        try:
            # Only call env_method if annealing is active to potentially reduce overhead
            if self.penalty_start != self.penalty_end and self.anneal_timesteps > 0:
                self.training_env.env_method('set_game_over_penalty', current_penalty)

                # Log current penalty coefficient less frequently for cleaner logs
                log_trigger = (self.num_timesteps % 5000 == 0) or \
                              (self.num_timesteps == 1) or \
                              (self.num_timesteps == self.anneal_timesteps + 1 and self.anneal_timesteps > 0)

                if log_trigger:
                    self.logger.record('train/current_go_penalty_coeff', current_penalty)
                    if self.verbose > 0:
                        # Use print to match original user code's callback output style
                        print(f" Timestep {self.num_timesteps}: GO Penalty updated to {current_penalty:.2f}")
        except AttributeError:
            # Log error only once
            if not self._callback_error_logged:
                if self.verbose > 0:
                    print("Warning: Could not call 'env_method' in CurriculumCallback. Update skipped.")
                self._callback_error_logged = True
        except Exception as e:
            # Log other errors only once
            if not self._callback_error_logged:
                if self.verbose > 0:
                     print(f"Warning: Error calling env_method in CurriculumCallback: {e}")
                self._callback_error_logged = True
        return True # Continue training

# ==============================================================================
# === Global Variables for Models & Envs ===
# ==============================================================================
model_p1 = model_p2 = model_p3 = None
train_env_p1 = train_env_p2 = train_env_p3 = None
phase1_success = phase2_success = phase3_success = False

# ==============================================================================
# === PHASE 1: Training Setup & Execution ===
# ==============================================================================
# <<< THIS BLOCK REPLACES THE ORIGINAL SINGLE ENV/MODEL/TRAINING SETUP >>>
# Properly formatted block
try:
    write_log(f"\n{'='*20} STARTING {PHASE_1_NAME} {'='*20}")

    # --- Environment Setup (Phase 1) ---
    write_log(f"Creating Env for {PHASE_1_NAME}...")
    def make_env_p1():
        return TetrisEnv(env_config=config_p1) # Pass P1 config
    train_env_p1_base = DummyVecEnv([make_env_p1])
    train_env_p1_stacked = VecFrameStack(train_env_p1_base, n_stack=config_p1["n_stack"], channels_order="first")
    train_env_p1 = VecNormalize(train_env_p1_stacked, norm_obs=False, norm_reward=True, gamma=config_p1["gamma"], clip_reward=10.0)
    write_log(f" P1 Env Created. Action Space: {train_env_p1.action_space}") # Should be 4

    # --- Callbacks (Phase 1) ---
    callback_list_p1 = []
    curriculum_cb_p1 = CurriculumCallback( # Inactive in P1
        penalty_start=config_p1["penalty_game_over_start_coeff"], penalty_end=config_p1["penalty_game_over_end_coeff"],
        anneal_fraction=config_p1["curriculum_anneal_fraction"], total_training_steps=config_p1["total_timesteps"], verbose=1)
    callback_list_p1.append(curriculum_cb_p1)
    if wandb_enabled and run:
        wandb_cb_p1 = WandbCallback(model_save_path=None, log="all", verbose=0) # Log to overall run
        callback_list_p1.append(wandb_cb_p1)
        write_log(" P1 CBs: Curric(Inactive), Wandb")
    else:
        write_log(" P1 CBs: Curric(Inactive)")

    # --- PPO Model (Phase 1 - New) ---
    write_log(f"Setting up NEW PPO Model for {PHASE_1_NAME}...")
    model_p1 = PPO( # Using parameters from config_p1
        policy=config_p1["policy_type"],
        env=train_env_p1,
        verbose=1,
        gamma=config_p1["gamma"],
        learning_rate=float(config_p1["learning_rate"]),
        n_steps=config_p1["n_steps"],
        batch_size=config_p1["batch_size"],
        n_epochs=config_p1["n_epochs"],
        gae_lambda=config_p1["gae_lambda"],
        clip_range=config_p1["clip_range"],
        ent_coef=config_p1["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"/kaggle/working/runs/{TOTAL_RUNTIME_ID}/Phase1" if wandb_enabled else None, # Subdir for TB
        policy_kwargs=dict(normalize_images=False)
    )
    # Configure logger for PPO model (handles Tensorboard and WandbCallback linkage)
    logger_p1 = configure_logger(
        tensorboard_log=model_p1.tensorboard_log,
        tb_log_name="PPO_P1", # Use specific TB log name
        reset_num_timesteps=True # Start timesteps from 0 for this phase
    )
    model_p1.set_logger(logger_p1)

    # --- Training (Phase 1) ---
    write_log(f"🚀 Starting P1 Training ({config_p1['total_timesteps']} steps)...")
    t_start_p1 = time.time()
    model_p1.learn(
        total_timesteps=config_p1["total_timesteps"],
        callback=callback_list_p1,
        log_interval=10, # Log every 10 updates
        reset_num_timesteps=True # Ensure timesteps reset for this .learn call
    )
    t_end_p1 = time.time()
    write_log(f"✅ P1 Complete! Time: {(t_end_p1 - t_start_p1) / 3600:.2f}h")

    # --- Saving (Phase 1) ---
    write_log(f"💾 Saving P1 model: {phase1_model_save_path}")
    model_p1.save(phase1_model_save_path)
    write_log(f"💾 Saving P1 VecNormalize stats: {phase1_stats_save_path}")
    train_env_p1.save(phase1_stats_save_path)
    phase1_success = True

except Exception as e:
    write_log(f"❌❌❌ P1 ERROR: {e}", True)
    phase1_success = False
except KeyboardInterrupt:
    write_log(f"🛑 P1 Training Interrupted by User.")
    phase1_success = False # Mark as unsuccessful if interrupted
finally:
    # Ensure Phase 1 environment is closed
    if train_env_p1 is not None:
        try:
            train_env_p1.close()
            write_log(" P1 Env closed.")
        except Exception as e:
            write_log(f" P1 Env close err: {e}")

# ==============================================================================
# === PHASE 2: Setup, Load & Execution ===
# ==============================================================================
# Properly formatted block
if phase1_success and os.path.exists(phase1_model_save_path) and os.path.exists(phase1_stats_save_path):
    try:
        write_log(f"\n{'='*20} STARTING {PHASE_2_NAME} {'='*20}")

        # --- Environment Setup (Phase 2) ---
        write_log(f"Creating Env for {PHASE_2_NAME}...")
        def make_env_p2():
            return TetrisEnv(env_config=config_p2) # Use P2 config
        train_env_p2_base = DummyVecEnv([make_env_p2])
        train_env_p2_stacked = VecFrameStack(train_env_p2_base, n_stack=config_p2["n_stack"], channels_order="first")
        # Load P1 Stats into P2 Env Wrapper
        write_log(f"🔄 Loading P1 VecNormalize stats: {phase1_stats_save_path}")
        train_env_p2 = VecNormalize.load(phase1_stats_save_path, train_env_p2_stacked)
        train_env_p2.training = True # Set to training mode
        write_log(f" P2 Env Created. Action Space: {train_env_p2.action_space}") # Should be 5

        # --- Callbacks (Phase 2) ---
        callback_list_p2 = []
        curriculum_cb_p2 = CurriculumCallback( # Active in P2
            penalty_start=config_p2["penalty_game_over_start_coeff"], penalty_end=config_p2["penalty_game_over_end_coeff"],
            anneal_fraction=config_p2["curriculum_anneal_fraction"], total_training_steps=config_p2["total_timesteps"], verbose=1)
        callback_list_p2.append(curriculum_cb_p2)
        if wandb_enabled and run:
            wandb_cb_p2 = WandbCallback(model_save_path=None, log="all", verbose=0)
            callback_list_p2.append(wandb_cb_p2)
            write_log(" P2 CBs: Curric(Active), Wandb")
        else:
            write_log(" P2 CBs: Curric(Active)")

        # --- Load P1 Model & Adapt (Phase 2) ---
        write_log(f"🔄 Loading P1 PPO model: {phase1_model_save_path}")
        model_p2 = PPO.load(
            phase1_model_save_path,
            env=train_env_p2, # CRITICAL: Use the new env instance
            device="cuda" if torch.cuda.is_available() else "cpu"
            # custom_objects can be used to update parameters, but we'll set them after load
        )
        write_log(" Model loaded. ⚠️ Action space changed 4->5!")

        # Set new parameters for the loaded model
        model_p2.set_env(train_env_p2) # Ensure correct env linkage
        model_p2.learning_rate = float(config_p2["learning_rate"]) # Update LR
        model_p2.ent_coef = float(config_p2["ent_coef"]) # Update Entropy
        # Clip range often needs a schedule function for SB3 PPO
        def clip_p2_fn(progress_remaining: float) -> float:
            # You could implement annealing here if needed, otherwise constant
            return float(config_p2["clip_range"])
        model_p2.clip_range = clip_p2_fn # Update Clip Range

        # Reset logger for Phase 2 (crucial for correct logging)
        model_p2.tensorboard_log = f"/kaggle/working/runs/{TOTAL_RUNTIME_ID}/Phase2" if wandb_enabled else None
        logger_p2 = configure_logger(
            tensorboard_log=model_p2.tensorboard_log,
            tb_log_name="PPO_P2", # Use specific TB log name
            reset_num_timesteps=False # DO NOT reset timesteps here
        )
        model_p2.set_logger(logger_p2)
        write_log(" P2 Model params updated and logger reset.")

        # --- Training (Phase 2) ---
        write_log(f"🚀 Starting P2 Training ({config_p2['total_timesteps']} steps)...")
        t_start_p2 = time.time()
        model_p2.learn(
            total_timesteps=config_p2["total_timesteps"],
            callback=callback_list_p2,
            log_interval=10,
            reset_num_timesteps=False # Continue timestep count
        )
        t_end_p2 = time.time()
        write_log(f"✅ P2 Complete! Time: {(t_end_p2 - t_start_p2) / 3600:.2f}h")

        # --- Saving (Phase 2 - Temp) ---
        write_log(f"💾 Saving P2 model: {phase2_model_save_path}")
        model_p2.save(phase2_model_save_path)
        write_log(f"💾 Saving P2 VecNormalize stats: {phase2_stats_save_path}")
        train_env_p2.save(phase2_stats_save_path)
        phase2_success = True

    except Exception as e:
        write_log(f"❌❌❌ P2 ERROR: {e}", True)
        phase2_success = False
    except KeyboardInterrupt:
        write_log(f"🛑 P2 Training Interrupted by User.")
        phase2_success = False
    finally: # Close P2 Env
        if train_env_p2 is not None:
            try:
                train_env_p2.close()
                write_log(" P2 Env closed.")
            except Exception as e:
                write_log(f" P2 Env close err: {e}")
else:
    write_log(f"⏩ Skipping {PHASE_2_NAME} due to Phase 1 failure or missing files.")


# ==============================================================================
# === PHASE 3: Setup, Load & Execution ===
# ==============================================================================
# Properly formatted block
if phase1_success and phase2_success and os.path.exists(phase2_model_save_path) and os.path.exists(phase2_stats_save_path):
    try:
        write_log(f"\n{'='*20} STARTING {PHASE_3_NAME} {'='*20}")

        # --- Env for Phase 3 (Using config_p3) ---
        write_log(f"Creating Env for {PHASE_3_NAME}...")
        def make_env_p3():
            return TetrisEnv(env_config=config_p3) # Use P3 config
        train_env_p3_base = DummyVecEnv([make_env_p3])
        train_env_p3_stacked = VecFrameStack(train_env_p3_base, n_stack=config_p3["n_stack"], channels_order="first")
        # Load P2 Stats into P3 Env Wrapper
        write_log(f"🔄 Loading P2 VecNormalize stats: {phase2_stats_save_path}")
        train_env_p3 = VecNormalize.load(phase2_stats_save_path, train_env_p3_stacked)
        train_env_p3.training = True
        write_log(f" P3 Env Created. Action Space: {train_env_p3.action_space}") # Should be 5

        # --- Callbacks (Phase 3) ---
        callback_list_p3 = []
        curriculum_cb_p3 = CurriculumCallback( # GO Penalty fixed in P3
            penalty_start=config_p3["penalty_game_over_start_coeff"], penalty_end=config_p3["penalty_game_over_end_coeff"],
            anneal_fraction=config_p3["curriculum_anneal_fraction"], total_training_steps=config_p3["total_timesteps"], verbose=1)
        callback_list_p3.append(curriculum_cb_p3)
        if wandb_enabled and run:
            wandb_cb_p3 = WandbCallback(model_save_path=None, log="all", verbose=0)
            callback_list_p3.append(wandb_cb_p3)
            write_log(" P3 CBs: Curric(Inactive), Wandb")
        else:
            write_log(" P3 CBs: Curric(Inactive)")

        # --- Load P2 Model & Adapt (Phase 3) ---
        write_log(f"🔄 Loading P2 PPO model: {phase2_model_save_path}")
        model_p3 = PPO.load(
            phase2_model_save_path,
            env=train_env_p3, # Use P3 env
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        write_log(" Model loaded.")

        # Set new parameters for P3
        model_p3.set_env(train_env_p3)
        model_p3.learning_rate = float(config_p3["learning_rate"])
        model_p3.ent_coef = float(config_p3["ent_coef"])
        def clip_p3_fn(p): return float(config_p3["clip_range"])
        model_p3.clip_range = clip_p3_fn

        # Reset logger for P3
        model_p3.tensorboard_log = f"/kaggle/working/runs/{TOTAL_RUNTIME_ID}/Phase3" if wandb_enabled else None
        logger_p3 = configure_logger(
            tensorboard_log=model_p3.tensorboard_log,
            tb_log_name="PPO_P3", # Use specific TB log name
            reset_num_timesteps=False # Continue timesteps
        )
        model_p3.set_logger(logger_p3)
        write_log(" P3 Model params updated and logger reset.")

        # --- Training (Phase 3) ---
        write_log(f"🚀 Starting P3 Training ({config_p3['total_timesteps']} steps)...")
        t_start_p3 = time.time()
        model_p3.learn(
            total_timesteps=config_p3["total_timesteps"],
            callback=callback_list_p3,
            log_interval=10,
            reset_num_timesteps=False # Continue timesteps
        )
        t_end_p3 = time.time()
        write_log(f"✅ P3 Complete! Time: {(t_end_p3 - t_start_p3) / 3600:.2f}h")

        # --- Saving (Phase 3 - FINAL) ---
        write_log(f"💾 Saving FINAL P3 model: {phase3_final_model_save_path}")
        model_p3.save(phase3_final_model_save_path)
        write_log(f"💾 Saving FINAL P3 stats: {phase3_final_stats_save_path}")
        train_env_p3.save(phase3_final_stats_save_path)
        # Display final file links
        display(FileLink(phase3_final_model_save_path))
        display(FileLink(phase3_final_stats_save_path))
        phase3_success = True

    except Exception as e:
        write_log(f"❌❌❌ P3 ERROR: {e}", True)
        phase3_success = False
    except KeyboardInterrupt:
        write_log(f"🛑 P3 Training Interrupted by User.")
        phase3_success = False
    finally: # Close P3 Env
        if train_env_p3 is not None:
            try:
                train_env_p3.close()
                write_log(" P3 Env closed.")
            except Exception as e:
                write_log(f" P3 Env close err: {e}")
else:
    write_log(f"⏩ Skipping {PHASE_3_NAME} due to previous phase failure or missing files.")


# --- Final Cleanup & Reporting ---
# Properly formatted block
write_log("🧹 Cleaning up final resources...")
if java_process and java_process.poll() is None:
    write_log(" Terminating Java server...")
    java_process.terminate()
    try:
        java_process.wait(timeout=5)
        write_log(" Java server terminated.")
    except subprocess.TimeoutExpired:
        java_process.kill()
        write_log(" Java server killed.")
    except Exception as e:
         write_log(f" Error terminating Java server: {e}")
elif java_process:
    write_log(" Java server already terminated.")
else:
    write_log(" Java server not started.")

# --- Upload final successful model to Wandb ---
final_model_to_upload = None
final_stats_to_upload = None
# Prioritize uploading the result of the latest successful phase
if phase3_success:
    final_model_to_upload = phase3_final_model_save_path
    final_stats_to_upload = phase3_final_stats_save_path
elif phase2_success:
    final_model_to_upload = phase2_model_save_path # Should be temp path, but use if P3 failed
    final_stats_to_upload = phase2_stats_save_path
elif phase1_success:
    final_model_to_upload = phase1_model_save_path
    final_stats_to_upload = phase1_stats_save_path

if wandb_enabled and run and final_model_to_upload and final_stats_to_upload:
    if os.path.exists(final_model_to_upload) and os.path.exists(final_stats_to_upload):
        write_log(f"   Uploading final successful artifacts ({os.path.basename(final_model_to_upload)}) to Wandb...")
        try:
            wandb.save(final_model_to_upload, base_path="/kaggle/working")
            wandb.save(final_stats_to_upload, base_path="/kaggle/working")
            write_log("   Upload successful.")
        except Exception as e:
            write_log(f"   ⚠️ Wandb upload error: {e}")
    else:
        write_log("   Final artifacts not found for Wandb upload.")
elif wandb_enabled and run:
    write_log("   Skipping final artifact upload as no phase completed successfully or files missing.")


# --- Finish Wandb Run ---
if wandb_enabled and run:
    # Define overall success as completing all 3 phases
    overall_success = phase1_success and phase2_success and phase3_success
    exit_code = 0 if overall_success else 1
    try:
        # Ensure run is still active before finishing
        if wandb.run and wandb.run.id == run.id:
            run.finish(exit_code=exit_code)
        write_log(f"   Wandb run '{TOTAL_RUNTIME_ID}' finished with exit code {exit_code}.")
    except Exception as finish_e:
        write_log(f"   ⚠️ Error finishing Wandb run: {finish_e}")

# --- Final Status Report ---
write_log(f"🏁🏁🏁 ALL PHASES ATTEMPTED ({TOTAL_RUNTIME_ID}) 🏁🏁🏁")
write_log(f"Phase 1 Success: {phase1_success}")
write_log(f"Phase 2 Success: {phase2_success}")
write_log(f"Phase 3 Success: {phase3_success}")

# --- Evaluation Section Removed ---
write_log("Evaluation section removed. Please run evaluation separately using the final saved model and stats.")