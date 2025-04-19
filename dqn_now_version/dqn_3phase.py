# -*- coding: utf-8 -*-
import numpy as np
import socket
import cv2
# import matplotlib.pyplot as plt # Matplotlib not strictly needed for core logic
import subprocess
import os
import shutil
import glob
import imageio
import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3.common.env_checker import check_env # Not used on VecEnv
from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env # Not used
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from IPython.display import FileLink, display # Image not used directly
# from stable_baselines3.common.callbacks import BaseCallback # Replaced by WandbCallback
import torch
import time
import pygame # Added for rendering in TetrisEnv
# from stable_baselines3 import PPO # PPO imported but not used, can be removed

# --- Wandb Setup ---
import os
import wandb
# Check if running in a Kaggle environment to use secrets
if 'KAGGLE_USERNAME' in os.environ:
    from kaggle_secrets import UserSecretsClient
    kaggle_env = True
else:
    kaggle_env = False
    print("Not running in a Kaggle environment. Skipping Kaggle Secrets.")


# Import WandbCallback for SB3 integration
from wandb.integration.sb3 import WandbCallback
def write_log(message, exc_info=False):
    """Appends a message to the log file and prints it."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
            if exc_info:
                 import traceback
                 traceback.print_exc(file=f)

    except Exception as e:
        print(f"Error writing to log file {log_path}: {e}")
    print(log_message)
    if exc_info:
        import traceback
        traceback.print_exc()
# --- Configuration ---
# Set your student ID here for filenames
STUDENT_ID = "YOUR_STUDENT_ID" # <<<<<<<<<<< 請修改為你的學號
# Set total training steps
TOTAL_TIMESTEPS = 2500000 # Adjust as needed (e.g., 1M, 2M, 5M) - Increased for potentially better results

# --- Wandb Login and Initialization ---
wandb_enabled = False
try:
    if kaggle_env:
        user_secrets = UserSecretsClient()
        WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.login()
        wandb_enabled = True
    else:
        # Attempt login directly if not in Kaggle (e.g., local machine with WANDB_API_KEY env var)
        if os.environ.get("WANDB_API_KEY"):
             wandb.login()
             wandb_enabled = True
        else:
             print("WANDB_API_KEY not found in environment variables.")
             wandb_enabled = False # Explicitly disable if key isn't set outside Kaggle


except Exception as e:
    print(f"Wandb login failed: {e}. Running without Wandb logging.")
    wandb_enabled = False


# Start a wandb run if enabled
# --- !!! MODIFY HYPERPARAMETERS HERE for Wandb logging if needed !!! ---
# These values will be used if not overridden by Wandb sweeps
config = { # Log hyperparameters
    "policy_type": "CnnPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1",
    "gamma": 0.99,
    "learning_rate": 2e-4,
    "buffer_size": 400000,
    "learning_starts": 10000, # Increased learning_starts slightly
    "target_update_interval": 1000, # MODIFIED: Reduced target update interval
    "train_freq": (1, "step"), # Train every step
    "gradient_steps": 1, # One gradient step per train_freq
    # --- MODIFIED: Increased exploration duration, slightly higher final eps ---
    "exploration_fraction": 0.5, # INCREASED exploration duration (e.g., 50% of training steps)
    "exploration_final_eps": 0.05, # Kept final exploration rate
    "batch_size": 32,
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- MODIFIED: Add reward coeffs to config for tracking AND adjusted values ---
    # QUADRATIC line clear bonus will be applied IN THE CODE using this coeff as base
    "reward_line_clear_base_coeff": 150.0, # Base reward for 1 line clear (e.g., 1*150)
    "penalty_height_increase_coeff": 5.0, # DECREASED penalty for height increase
    "penalty_hole_increase_coeff": 10.0, # DECREASED penalty for hole increase
    "penalty_step_coeff": 0.0, # SET TO ZERO - Removed survival penalty
    "penalty_game_over_coeff": 200.0 # Slightly increased game over penalty
}

run = None # Initialize run to None
run_id = f"local_{int(time.time())}" # Default local ID

if wandb_enabled:
    try:
        run = wandb.init(
            project="tetris-training-improved", # <<<<<<<<<<< 可修改專案名稱
            entity="YOUR_WANDB_ENTITY", # <<<<<<<<<<< 請修改為你的 Wandb entity (例如你的用戶名)
            sync_tensorboard=True,
            monitor_gym=True, # Automatically log gym environment stats
            save_code=True,
            config=config # Log hyperparameters from the dictionary
        )
        run_id = run.id # Get run ID for saving paths
        write_log(f"✅ Wandb run initialized: {run.url}")
    except Exception as e:
        write_log(f"❌ Failed to initialize Wandb run: {e}. Running without Wandb logging features.")
        run = None # Ensure run is None if initialization fails
        wandb_enabled = False # Disable wandb features


log_path = f"/kaggle/working/tetris_train_log_{run_id}.txt"

def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60):
    """Waits for the Tetris TCP server to become available."""
    write_log(f"⏳ 等待 Tetris TCP server 啟動中 ({ip}:{port})...")
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((ip, port), timeout=1.0): # Use create_connection for a more robust check
                pass # Connection successful
            write_log("✅ Java TCP server 準備完成，連線成功")
            return True # Indicate success
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            if time.time() - start_time > timeout:
                write_log(f"❌ 等待 Java TCP server 超時 ({timeout}s): {e}")
                return False # Indicate failure
            # write_log(f"    連接失敗 ({e}), 等待重試...") # Too noisy
            time.sleep(1.0) # Wait a bit longer before retrying
        except Exception as e:
            write_log(f"❌ 等待 Java TCP server 時發生未知錯誤: {e}", exc_info=True)
            if time.time() - start_time > timeout:
                 return False
            time.sleep(1.0)


# --- Start Java Server ---
java_process = None # Initialize to None
try:
    write_log("🚀 嘗試啟動 Java Tetris server...")
    jar_file = "TetrisTCPserver_v0.6.jar" # <<<<<<<<<<< 請確認 JAR 檔案名稱及路徑
    if not os.path.exists(jar_file):
         write_log(f"❌ 錯誤: 找不到 JAR 檔案 '{jar_file}'。請確保它在工作目錄中。")
         raise FileNotFoundError(f"JAR file '{jar_file}' not found.")

    # Start process, redirect stdout/stderr to DEVNULL to keep console clean
    # shell=True might be needed on some OS/environments, but can be risky
    # Consider removing shell=True if not necessary
    java_process = subprocess.Popen(
        ["java", "-jar", jar_file],
        stdout=subprocess.DEVNULL, # Hide server stdout
        stderr=subprocess.DEVNULL, # Hide server stderr
        # shell=True # Optional: use shell if needed for path resolution
    )
    write_log(f"✅ Java server process 啟動 (PID: {java_process.pid})")
    if not wait_for_tetris_server():
        raise TimeoutError("Java server did not become available within the timeout.") # Raise specific error

except Exception as e:
    write_log(f"❌ 啟動或等待 Java server 時發生錯誤: {e}", exc_info=True)
    # Attempt to terminate if process started but failed connection
    if java_process and java_process.poll() is None:
         write_log("    嘗試終止未成功連接的 Java server process...")
         java_process.terminate()
         try:
             java_process.wait(timeout=5)
         except subprocess.TimeoutExpired:
             write_log("    Java server 未能在 5 秒內終止, 強制結束...")
             java_process.kill()
    # If wandb is enabled and running, finish it with an error code
    if run and hasattr(run, 'is_running') and run.is_running:
         run.finish(exit_code=1, quiet=True)
    raise # Re-raise the exception to stop the script

# --- Check GPU ---
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    write_log(f"✅ PyTorch is using GPU: {device_name}")
    device = "cuda"
else:
    write_log("⚠️ PyTorch is using CPU. Training will be significantly slower.")
    device = "cpu"


# ----------------------------
# 定義 Tetris 環境 (採用老師的格式, 結合獎勵機制概念)
# ----------------------------
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris that interacts with a Java TCP server."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS = 5
    IMG_HEIGHT = 200 # Original server image height
    IMG_WIDTH = 100  # Original server image width
    IMG_CHANNELS = 3 # Original server image channels (BGR)
    RESIZED_DIM = 84 # Target dimension for observation space

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        # Observation space is grayscale, channel first
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), # (Channels, Height, Width) - Grayscale, channel first
            dtype=np.uint8
        )
        self.server_ip = host_ip
        self.server_port = host_port
        self.client_sock = None
        # Connect in init - added retry logic here too
        self._connect_socket(retries=5, delay=1.0)


        # Reward shaping & statistics variables
        self.lines_removed = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        # Store last observation in case of error during receive
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        # Store last raw frame for rendering
        self.last_raw_render_frame = np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)


        # --- !!! MODIFIED: REWARD SHAPING COEFFICIENTS !!! ---
        # Retrieve from Wandb config if available, otherwise use defaults from global config dict
        # Using .get() with default ensures it works even if config changes
        current_config = run.config if run else config # Use global config if no run
        self.reward_line_clear_base_coeff = current_config.get("reward_line_clear_base_coeff", 150.0) # Base reward for 1 line
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 5.0) # Penalty per unit height increase
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 10.0) # Penalty per unit hole increase
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.0) # Penalty per step (should be 0.0)
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_coeff", 200.0) # Penalty for game over

        write_log(f"TetrisEnv initialized with Reward Coeffs: LineBase={self.reward_line_clear_base_coeff}, H_Inc={self.penalty_height_increase_coeff}, O_Inc={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}")


        # For rendering
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False # Track Pygame init state
        # Flag to prevent Wandb log error spam
        self._wandb_log_error_reported = False
        # Flag for render mode issue log
        self._eval_render_mode_error_reported = False
        self._eval_render_error_reported = False
        self._eval_render_error_reported_access = False


    def _initialize_pygame(self):
        """Initializes Pygame if not already done."""
        # Only initialize if render_mode is human and not already initialized
        if self.render_mode == "human" and not self.is_pygame_initialized:
            try:
                import pygame
                pygame.init()
                pygame.display.init()
                # Scale window for better visibility (e.g., 4x the resized dimension)
                window_width = self.RESIZED_DIM * 4
                window_height = self.RESIZED_DIM * 4
                self.window_surface = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log("    Pygame initialized for rendering (human mode).")
            except ImportError:
                write_log("⚠️ Pygame not installed, cannot use 'human' render mode.")
                self.render_mode = None # Disable human rendering
            except Exception as e:
                write_log(f"⚠️ Error initializing Pygame: {e}")
                self.render_mode = None # Disable human rendering

    def _connect_socket(self, retries=1, delay=0.1):
        """Establishes connection to the game server with retries."""
        if self.client_sock:
            try:
                self.client_sock.close()
            except socket.error:
                pass # Ignore error on closing
            self.client_sock = None

        for attempt in range(retries):
            try:
                self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_sock.settimeout(10.0) # Set timeout for connection attempt
                self.client_sock.connect((self.server_ip, self.server_port))
                self.client_sock.settimeout(5.0) # Reset to a shorter timeout for subsequent operations
                # write_log(f"🔌 Socket connected to {self.server_ip}:{self.server_port} on attempt {attempt+1}") # Less verbose
                return # Connection successful

            except socket.error as e:
                if attempt < retries - 1:
                    # write_log(f"    Connection attempt {attempt+1} failed: {e}. Retrying in {delay}s.") # Too noisy
                    time.sleep(delay)
                else:
                    write_log(f"❌ Final socket connection attempt failed after {retries} retries: {e}", exc_info=True)
                    self.client_sock = None # Ensure sock is None on failure
                    raise ConnectionError(f"Failed to connect to Tetris server at {self.server_ip}:{self.server_port}") from e
            except Exception as e:
                write_log(f"❌ Unexpected error during connection attempt {attempt+1}: {e}", exc_info=True)
                if attempt < retries - 1:
                     time.sleep(delay)
                else:
                    self.client_sock = None
                    raise ConnectionError(f"Unexpected error connecting to Tetris server: {e}") from e


    def _send_command(self, command: bytes):
        """Sends a command to the server, handles potential errors."""
        if not self.client_sock:
             # Attempt to reconnect if socket is unexpectedly closed
             write_log("⚠️ Socket not connected during send. Attempting reconnect...")
             try:
                 self._connect_socket(retries=3, delay=0.5)
                 write_log("    Reconnect successful.")
             except ConnectionError as e:
                 write_log(f"    Reconnect failed: {e}")
                 raise ConnectionError("Socket is not connected and reconnect failed. Cannot send command.") from e

        try:
            self.client_sock.sendall(command)
        except socket.timeout:
            write_log("❌ Socket timeout during send.")
            self.client_sock = None # Mark socket as bad
            raise ConnectionAbortedError("Socket timeout during send")
        except socket.error as e:
            write_log(f"❌ Socket error during send: {e}")
            self.client_sock = None # Mark socket as bad
            raise ConnectionAbortedError(f"Socket error during send: {e}")

    def _receive_data(self, size):
        """Receives exactly size bytes from the server."""
        if not self.client_sock:
             # Attempt to reconnect if socket is unexpectedly closed
             write_log("⚠️ Socket not connected during receive. Attempting reconnect...")
             try:
                 self._connect_socket(retries=3, delay=0.5)
                 write_log("    Reconnect successful.")
             except ConnectionError as e:
                 write_log(f"    Reconnect failed: {e}")
                 raise ConnectionError("Socket is not connected and reconnect failed. Cannot receive data.") from e


        data = b""
        try:
            self.client_sock.settimeout(5.0) # Set timeout for recv (shorter than send)
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    write_log("❌ Socket connection broken during receive (received empty chunk).")
                    self.client_sock = None # Mark socket as bad
                    raise ConnectionAbortedError("Socket connection broken")
                data += chunk
        except socket.timeout:
             write_log(f"❌ Socket timeout during receive (expected {size}, got {len(data)}).")
             self.client_sock = None # Mark socket as bad
             raise ConnectionAbortedError("Socket timeout during receive")
        except socket.error as e:
            write_log(f"❌ Socket error during receive: {e}")
            self.client_sock = None # Mark socket as bad
            raise ConnectionAbortedError(f"Socket error during receive: {e}")
        except Exception as e:
            write_log(f"❌ Unexpected error during receive: {e}", exc_info=True)
            self.client_sock = None
            raise ConnectionAbortedError(f"Unexpected error during receive: {e}")

        return data

    def get_tetris_server_response(self):
        """Gets state update from the Tetris server via socket."""
        try:
            # Read game over byte (1 byte)
            is_game_over_byte = self._receive_data(1)
            is_game_over = (is_game_over_byte == b'\x01')

            # Read stats (4 bytes each for lines, height, holes)
            removed_lines_bytes = self._receive_data(4)
            removed_lines = int.from_bytes(removed_lines_bytes, 'big')

            height_bytes = self._receive_data(4)
            height = int.from_bytes(height_bytes, 'big')

            holes_bytes = self._receive_data(4)
            holes = int.from_bytes(holes_bytes, 'big')

            # Read image size (4 bytes)
            img_size_bytes = self._receive_data(4)
            img_size = int.from_bytes(img_size_bytes, 'big')

            # Check image size validity
            if img_size <= 0 or img_size > 2000000: # Increased max size for safety
                 write_log(f"❌ Received invalid image size: {img_size}. Aborting receive.")
                 # Return last known state and signal termination
                 # Ensure socket is marked bad so subsequent steps fail fast
                 self.client_sock = None
                 return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

            # Read image data (img_size bytes)
            img_png = self._receive_data(img_size)

            # Decode and preprocess image
            nparr = np.frombuffer(img_png, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if np_image is None:
                 write_log("❌ Failed to decode image from server response.")
                 # Return last known state and signal termination
                 self.client_sock = None
                 return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

            # Resize and convert to grayscale
            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Add channel dimension (channel first)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8)

            # Store frames for rendering/observation
            self.last_raw_render_frame = resized.copy() # Store BGR for render
            self.last_observation = observation.copy() # Store processed obs

            return is_game_over, removed_lines, height, holes, observation

        except (ConnectionAbortedError, ConnectionRefusedError, ValueError, socket.error) as e:
             write_log(f"❌ Connection/Value error getting server response: {e}. Ending episode.")
             self.client_sock = None # Mark socket as bad on known errors
             # Return last known state and signal termination
             return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"❌ Unexpected error getting server response: {e}. Ending episode.", exc_info=True)
            self.client_sock = None # Mark socket as bad on unexpected errors
            # Return last known state and signal termination
            return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()


    def step(self, action):
        # --- Send Action ---
        command_map = {
            0: b"move -1\n", # Left
            1: b"move 1\n",  # Right
            2: b"rotate 0\n", # Rotate Left
            3: b"rotate 1\n", # Rotate Right
            4: b"drop\n"     # Drop
        }
        command = command_map.get(action)
        if command is None:
            write_log(f"⚠️ Invalid action received: {action}. Sending 'drop'.")
            command = b"drop\n"

        # write_log(f"Step {self.lifetime + 1}: Chosen Action={action}, Command={command.strip()}") # Too verbose usually

        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            write_log(f"❌ Ending episode due to send failure in step: {e}", exc_info=True)
            terminated = True
            observation = self.last_observation.copy() # Return last valid observation
            reward = self.penalty_game_over_coeff * -1 # Apply game over penalty directly
            info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'send_error'}
            info['terminal_observation'] = observation.copy() # Add terminal observation

            # --- Log detailed rewards on send failure termination ---
            if wandb_enabled and run:
                try:
                    # Log zero for reward components except the game over penalty
                    log_data = {
                        "reward/step_total": reward,
                        "reward/step_line_clear": 0.0,
                        "reward/step_height_penalty": 0.0,
                        "reward/step_hole_penalty": 0.0,
                        "reward/step_survival_penalty": 0.0,
                        "reward/step_game_over_penalty": -self.penalty_game_over_coeff, # Log the penalty
                        "env/lines_cleared_this_step": 0,
                        "env/height_increase": 0, # We don't know the increase without response
                        "env/hole_increase": 0,   # We don't know the increase without response
                        "env/current_height": self.current_height, # Log last known state
                        "env/current_holes": self.current_holes,   # Log last known state
                        "env/current_lifetime": self.lifetime
                    }
                    wandb.log(log_data) # Log immediately
                except Exception as log_e:
                     if not self._wandb_log_error_reported:
                         write_log(f"Wandb logging error in step (send fail): {log_e}")
                         self._wandb_log_error_reported = True
            # --- End logging ---

            return observation, reward, terminated, False, info # Return immediately

        # --- Get State Update ---
        # This call also handles communication errors and returns terminated=True in that case
        terminated, new_lines_removed, new_height, new_holes, observation = self.get_tetris_server_response()

        # If receive failed (returns terminated=True but might not be GO on server),
        # it's treated as a terminal state due to communication error.
        # Check if terminated is true AND stats didn't change AND observation is the same (implies communication failed)
        if terminated and (new_lines_removed == self.lines_removed and new_height == self.current_height and new_holes == self.current_holes and np.array_equal(observation, self.last_observation)):
             # The get_tetris_server_response already logged the error, just finalize step
             reward = self.penalty_game_over_coeff * -1 # Ensure game over penalty
             info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'receive_error'}
             info['terminal_observation'] = observation.copy()
             # Log this specific termination case breakdown if receive error didn't already log it fully
             # (get_tetris_server_response logs the error message but not the reward breakdown)
             if wandb_enabled and run:
                 try:
                      log_data = {
                          "reward/step_total": reward,
                          "reward/step_line_clear": 0.0,
                          "reward/step_height_penalty": 0.0,
                          "reward/step_hole_penalty": 0.0,
                          "reward/step_survival_penalty": 0.0,
                          "reward/step_game_over_penalty": -self.penalty_game_over_coeff, # Log the penalty
                          "env/lines_cleared_this_step": 0,
                          "env/height_increase": 0,
                          "env/hole_increase": 0,
                          "env/current_height": self.current_height,
                          "env/current_holes": self.current_holes,
                          "env/current_lifetime": self.lifetime
                      }
                      wandb.log(log_data)
                 except Exception as log_e:
                      if not self._wandb_log_error_reported:
                           write_log(f"Wandb logging error in step (receive fail breakdown): {log_e}")
                           self._wandb_log_error_reported = True
             return observation, reward, terminated, False, info


        # --- Calculate Reward (if receive was successful) ---
        reward = 0.0
        lines_cleared_this_step = new_lines_removed - self.lines_removed

        # --- !!! MODIFIED: Multi-line clear reward logic (Quadratic Bonus) !!! ---
        line_clear_reward = 0.0
        if lines_cleared_this_step > 0:
            # Use the base coefficient and apply quadratic scaling
            if lines_cleared_this_step == 1:
                line_clear_reward = 1 * self.reward_line_clear_base_coeff
            elif lines_cleared_this_step == 2:
                line_clear_reward = 4 * self.reward_line_clear_base_coeff # Double line -> 4x base
            elif lines_cleared_this_step == 3:
                line_clear_reward = 9 * self.reward_line_clear_base_coeff # Triple line -> 9x base
            elif lines_cleared_this_step >= 4: # Tetris or more (should only be Tetris)
                line_clear_reward = 25 * self.reward_line_clear_base_coeff # Tetris -> 25x base (significant bonus)
            # Add a small bonus for any line clear to differentiate from no clear
            reward += line_clear_reward
        # --- END MODIFIED ---


        height_increase = new_height - self.current_height
        height_penalty = 0.0
        # Only penalize if height increased
        if height_increase > 0:
            height_penalty = height_increase * self.penalty_height_increase_coeff
            reward -= height_penalty
        # Optional: Reward height decrease slightly? (Not implemented here, keeping simple)
        # elif height_increase < 0:
        #     reward += abs(height_increase) * some_small_positive_coeff

        hole_increase = new_holes - self.current_holes
        hole_penalty = 0.0
        # Only penalize if holes increased
        if hole_increase > 0:
            hole_penalty = hole_increase * self.penalty_hole_increase_coeff
            reward -= hole_penalty
        # Optional: Reward hole decrease slightly? (Not implemented here, keeping simple)
        # elif hole_increase < 0:
        #     reward += abs(hole_increase) * some_small_positive_coeff


        step_penalty = self.penalty_step_coeff # This is 0.0 from config/default
        reward -= step_penalty # Apply step penalty (will be 0)

        game_over_penalty = 0.0
        # Only apply game over penalty if server explicitly reported game over (not a communication error)
        if terminated and (new_lines_removed != self.lines_removed or new_height != self.current_height or new_holes != self.current_holes or not np.array_equal(observation, self.last_observation)):
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            # Log only once per game over for clarity, ADDED reward breakdown
            write_log(f"💔 Game Over! Final Lines: {new_lines_removed}, Lifetime: {self.lifetime + 1}. Step Reward Breakdown: LC={line_clear_reward:.2f}, HP={-height_penalty:.2f}, OP={-hole_penalty:.2f}, SP={-step_penalty:.2f}, GO={-game_over_penalty:.2f} -> Total={reward:.2f}")

        # --- Update Internal State ---
        self.lines_removed = new_lines_removed
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        # Gym requires info to be a dict, even if empty
        info = {
            'removed_lines': self.lines_removed, # Total lines cleared in this episode
            'lifetime': self.lifetime, # Total steps in this episode
            'lines_cleared_this_step': lines_cleared_this_step, # Lines cleared by THIS action
            'height_increase': height_increase, # Change in height this step
            'hole_increase': hole_increase # Change in holes this step
        }

        # Ensure terminal_observation is added only if terminated
        if terminated:
             info['terminal_observation'] = observation.copy()
             # Add a final status if it's a server-side game over
             if game_over_penalty > 0:
                 info['final_status'] = 'game_over'


        truncated = False # DQN typically doesn't use truncation like PPO

        # --- !!! NEW: Detailed Wandb Logging !!! ---
        if wandb_enabled and run:
             try:
                 log_data = {
                     "reward/step_total": reward,
                     "reward/step_line_clear": line_clear_reward,
                     "reward/step_height_penalty": -height_penalty, # Log penalties as negative values
                     "reward/step_hole_penalty": -hole_penalty,
                     "reward/step_survival_penalty": -step_penalty, # Will be 0
                     "reward/step_game_over_penalty": -game_over_penalty, # Will be non-zero only on last step
                     "env/lines_cleared_this_step": lines_cleared_this_step,
                     "env/height_increase": height_increase,
                     "env/hole_increase": hole_increase,
                     "env/current_height": self.current_height,
                     "env/current_holes": self.current_holes,
                     "env/current_lifetime": self.lifetime # Log lifetime at each step
                 }
                 # Filter out zero reward components (except game over) for cleaner graphs in Wandb
                 # Keep all env/ metrics
                 # Only log non-zero rewards or all env metrics
                 filtered_log_data = {k: v for k, v in log_data.items() if not (k.startswith("reward/") and not k.endswith("game_over_penalty") and v == 0) or k.startswith("env/")}
                 # Use the global step provided by the SB3 callback implicitly
                 wandb.log(filtered_log_data)
             except Exception as log_e:
                 # Prevent spamming logs if Wandb logging fails repeatedly
                 if not self._wandb_log_error_reported:
                     write_log(f"Wandb logging error in step: {log_e}")
                     self._wandb_log_error_reported = True
        # --- END NEW ---


        # Optional: Render on step if requested
        if self.render_mode == "human":
             self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need to call super().reset(seed=seed) first for Gym API compliance
        # even if we don't strictly use the seed here in the server interaction.
        super().reset(seed=seed)
        # Reset the Wandb error reported flags for the new episode
        self._wandb_log_error_reported = False
        self._eval_render_mode_error_reported = False
        self._eval_render_error_reported = False
        self._eval_render_error_reported_access = False


        # Ensure socket is connected before sending reset command
        if not self.client_sock:
            write_log("⚠️ Socket not connected on reset. Attempting to reconnect...")
            try:
                self._connect_socket(retries=3, delay=0.5)
                write_log("    Reconnect successful during reset.")
            except ConnectionError as e:
                write_log(f"❌ Fatal: Reconnect failed during reset: {e}. Cannot start new episode.")
                # If wandb is enabled and running, finish it with an error code
                if run and hasattr(run, 'is_running') and run.is_running:
                    run.finish(exit_code=1, quiet=True)
                raise RuntimeError(f"Failed to reconnect to Tetris server on reset: {e}") from e


        for attempt in range(5): # Allow more attempts to reset/reconnect
            try:
                # Ensure the socket is still healthy before sending start
                self.client_sock.settimeout(5.0) # Short timeout for sending start
                self._send_command(b"start\n")
                # Give server a moment to process start and generate initial state
                time.sleep(0.2) # Increased delay slightly
                terminated, lines, height, holes, observation = self.get_tetris_server_response()

                # If get_tetris_server_response failed due to communication error,
                # it would return terminated=True and last observation.
                # We need to differentiate server-side game over from communication failure on reset.
                # A server-side game over right after 'start' is unlikely unless the server is misconfigured
                # or the previous game didn't truly end.
                # Let's check if the observation looks like a valid starting state (not all black/same as last)
                # This is a heuristic check. A better way depends on server specific state info.
                is_likely_communication_error = terminated and np.array_equal(observation, self.last_observation)

                if terminated and not is_likely_communication_error:
                    write_log(f"⚠️ Server reported game over on reset attempt {attempt+1}. Retrying...")
                    if attempt < 4: # Reconnect and retry if not last attempt
                         self._connect_socket(retries=3, delay=0.5) # Reconnect ensures a fresh state if server closed connection
                         time.sleep(1.0) # Small delay before retry
                         continue # Retry the loop
                    else:
                        write_log("❌ Server still terminated after multiple reset attempts. Cannot proceed.")
                        raise RuntimeError("Tetris server failed to reset properly.")
                elif is_likely_communication_error:
                     write_log(f"⚠️ Communication error getting initial state on reset attempt {attempt+1}. Retrying...")
                     if attempt < 4:
                          self._connect_socket(retries=3, delay=0.5)
                          time.sleep(1.0)
                          continue
                     else:
                          write_log("❌ Failed to get valid initial state after multiple communication error on reset attempts. Cannot proceed.")
                          raise RuntimeError("Tetris server failed to return initial state properly.")

                # Reset successful (terminated is False and no communication error detected)
                self.lines_removed = 0 # Reset internal counters for the new episode
                self.current_height = height
                self.current_holes = holes
                self.lifetime = 0
                self.last_observation = observation.copy()
                # write_log(f"🔄 Environment Reset. Initial state: H={height}, O={holes}") # Less verbose logging
                info = {} # Info dict for reset is usually empty or contains initial state info
                self.client_sock.settimeout(5.0) # Reset timeout for step operations
                return observation, info

            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError) as e:
                 write_log(f"🔌 Connection issue during reset attempt {attempt+1} ({e}). Retrying...", exc_info=True)
                 if attempt < 4:
                     try:
                         self._connect_socket(retries=3, delay=0.5) # Attempt reconnect
                         time.sleep(1.0)
                     except ConnectionError:
                         write_log("    Reconnect failed during retry.")
                         if attempt == 3: # If second to last attempt also fails, raise
                              raise RuntimeError(f"Failed to reconnect and reset Tetris server after multiple attempts: {e}") from e
                 else: # Final attempt failed
                     raise RuntimeError(f"Failed to reset Tetris server after multiple attempts: {e}") from e
            except Exception as e:
                 write_log(f"❌ Unexpected error during reset attempt {attempt+1}: {e}. Retrying...", exc_info=True)
                 if attempt < 4:
                     time.sleep(1.0)
                     continue
                 else:
                     raise RuntimeError(f"Failed to reset Tetris server after multiple attempts due to unexpected error: {e}") from e


        # Should not be reached if logic is correct, but as fallback:
        # If loop finishes without returning, it means all attempts failed
        raise RuntimeError("Failed to reset Tetris server after exhausting retry attempts.")


    def render(self):
        """Renders the environment."""
        # Ensure pygame is ready if in human mode
        self._initialize_pygame()

        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None:
                 write_log("⚠️ Render called but Pygame window is not initialized.")
                 return

            # Check if we have a frame to render
            if hasattr(self, 'last_raw_render_frame') and self.last_raw_render_frame is not None and self.last_raw_render_frame.shape[2] == 3:
                try:
                    # last_raw_render_frame is (H, W, C) BGR from OpenCV
                    render_frame_rgb = cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
                    # Pygame surface requires (width, height)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    # Transpose needed: (H, W, C) -> (W, H, C) for Pygame surfarray
                    # surfarray.blit_array expects shape (width, height, channels)
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))
                    # Scale up to window size
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0))
                    pygame.event.pump() # Process internal Pygame events
                    pygame.display.flip() # Update the full screen surface
                    if self.clock: # Ensure clock exists before ticking
                        self.clock.tick(self.metadata["render_fps"]) # Control frame rate
                except Exception as e:
                    write_log(f"⚠️ Error during Pygame rendering: {e}", exc_info=True)
                    # Optionally disable rendering on error: self.render_mode = None

            else:
                # Draw a black screen if no frame available yet or frame is invalid
                 self.window_surface.fill((0, 0, 0))
                 pygame.display.flip()

        elif self.render_mode == "rgb_array":
             # Return RGB (H, W, C) array
             if hasattr(self, 'last_raw_render_frame') and self.last_raw_render_frame is not None and self.last_raw_render_frame.shape[2] == 3:
                 return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
             else:
                 # Return black frame if no observation yet or frame is invalid
                 # Ensure the shape matches what is expected for rgb_array
                 return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

        else:
            # Rendering is disabled or not supported for the given mode
            pass


    def close(self):
        """Closes the environment, including socket and pygame."""
        # write_log("🔌 Closing environment connection.") # Less verbose
        if self.client_sock:
            try:
                # Optionally send a quit command to the server
                # Depends on server support for a 'quit' command
                # self._send_command(b"quit\n")
                self.client_sock.close()
            except socket.error as e:
                 write_log(f"    Error closing socket: {e}")
            self.client_sock = None

        if self.is_pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.is_pygame_initialized = False
                # write_log("    Pygame window closed.") # Less verbose
            except Exception as e:
                 write_log(f"    Error closing Pygame: {e}")

# --- Environment Setup ---
write_log("✅ 建立基礎環境函數 make_env...")
def make_env():
    """Helper function to create an instance of the Tetris environment."""
    # Pass render_mode to the environment constructor if needed for evaluation later
    # For the training env, render_mode is typically None or "rgb_array" if you log videos during training
    # For eval env later, we'll explicitly set "rgb_array" for GIF
    env = TetrisEnv()
    return env

write_log("✅ 建立向量化環境 (DummyVecEnv)...")
# Use DummyVecEnv for single environment interaction
train_env_base = DummyVecEnv([make_env])

write_log("✅ 包裝環境 (VecFrameStack)...")
# Wrap with VecFrameStack (channel-first is important for PyTorch CNNs)
# Use wandb config if available, otherwise use default from global config
n_stack = run.config.get("n_stack", config["n_stack"]) if run else config["n_stack"]
train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack, channels_order="first")
write_log(f"    已設定 FrameStack 數量: {n_stack}")

write_log("✅ 包裝環境 (VecNormalize - Rewards Only)...")
# Wrap with VecNormalize, NORMALIZING REWARDS ONLY.
# This helps stabilize training with potentially large shaped rewards.
# gamma should match the model's gamma for correct reward normalization
gamma_param = run.config.get("gamma", config["gamma"]) if run else config["gamma"]
train_env = VecNormalize(train_env_stacked, norm_obs=False, norm_reward=True, gamma=gamma_param)
write_log(f"    VecNormalize 設定: norm_obs=False, norm_reward=True, gamma={gamma_param}")


write_log("    環境建立完成並已包裝 (DummyVecEnv -> VecFrameStack -> VecNormalize)")


# ----------------------------
# DQN Model Setup and Training
# ----------------------------
write_log("🧠 設定 DQN 模型...")
# Use wandb config for hyperparameters if available, otherwise use defaults from global config dict
current_config = run.config if run else config # Use global config if no run active

policy_type = current_config.get("policy_type", "CnnPolicy")
learning_rate = current_config.get("learning_rate", 1e-4)
buffer_size = current_config.get("buffer_size", 100000)
learning_starts = current_config.get("learning_starts", 10000)
batch_size = current_config.get("batch_size", 32)
tau = 1.0 # Default for DQN target network update rate (1.0 means hard update)
target_update_interval = current_config.get("target_update_interval", 1000) # MODIFIED: Reduced target update interval
train_freq = current_config.get("train_freq", (1, "step")) # Default (1, "step")
gradient_steps = current_config.get("gradient_steps", 1) # Default 1
# --- !!! UPDATED Exploration Fraction used here !!! ---
exploration_fraction = current_config.get("exploration_fraction", 0.5) # INCREASED default if not in wandb
exploration_final_eps = current_config.get("exploration_final_eps", 0.05)
# Use the determined device (cuda or cpu)
device = torch.device(device)


# Define DQN model
model = DQN(
    policy=policy_type,
    env=train_env,
    verbose=1, # Set to 1 or 2 for training logs
    gamma=gamma_param, # Use loaded gamma from config/wandb
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    tau=tau,
    train_freq=train_freq, # Use from config
    gradient_steps=gradient_steps, # Use from config
    target_update_interval=target_update_interval, # Use from config
    exploration_fraction=exploration_fraction, # Use the updated value from config/wandb
    exploration_final_eps=exploration_final_eps, # Use the updated value from config/wandb
    # policy_kwargs is where you'd configure custom networks or potentially NoisyNet if supported
    # For standard CnnPolicy, normalize_images=False is common as VecFrameStack handles uint8 to float
    policy_kwargs=dict(normalize_images=False), # As per original code, let VecNormalize handle normalization
    seed=42, # Set seed for reproducibility
    device=device, # Use the determined device
    # Log TensorBoard data to a directory that Wandb can sync from
    tensorboard_log=f"/kaggle/working/runs/{run_id}" if wandb_enabled else None
)
write_log(f"    模型建立完成. Device: {model.device}")
# write_log(f"    使用的超參數: {model.get_parameters()['policy']}") # Optional: log policy network details


# Setup Wandb callback if enabled
if wandb_enabled:
    # Ensure the model save path exists
    model_save_dir = f"/kaggle/working/models/{run_id}"
    os.makedirs(model_save_dir, exist_ok=True)
    write_log(f"    模型將儲存至: {model_save_dir}")

    wandb_callback = WandbCallback(
        gradient_save_freq=10000, # Log grads every 10k steps
        model_save_path=model_save_dir, # Save models periodically
        model_save_freq=100000, # Save every 100k steps
        log="all", # Log histograms, gradients, etc.
        verbose=2,
        # Optionally log VecNormalize stats periodically
        # sync_vecnormalize=True # This is the default behavior if VecNormalize is detected
    )
    callback_list = [wandb_callback]
else:
    callback_list = None # No callback if wandb is disabled


# --- Training ---
write_log(f"🚀 開始訓練 {TOTAL_TIMESTEPS} 步...")
training_successful = False
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        log_interval=100 # Log basic stats (like FPS, mean reward) more frequently
    )
    write_log("✅ 訓練完成!")
    training_successful = True
except Exception as e:
    write_log(f"❌ 訓練過程中發生錯誤: {e}", exc_info=True) # Log exception info with traceback
    # Save model before exiting if error occurs mid-training
    error_save_path = f'/kaggle/working/{STUDENT_ID}_dqn_error_save_{run_id}.zip' # Include run_id
    write_log(f"    訓練中斷, 嘗試儲存模型至 {error_save_path}")
    try:
        # Ensure the model actually exists and has learned something before saving
        if hasattr(model, 'num_timesteps') and model.num_timesteps > 0:
             model.save(error_save_path)
             write_log(f"    模型已嘗試儲存至 {error_save_path}")
             if wandb_enabled and run: wandb.save(error_save_path) # Upload error model to wandb
        else:
             write_log("    模型尚未開始訓練 (num_timesteps is 0), 無法儲存.")
    except Exception as save_e:
         write_log(f"    ❌ 儲存錯誤模型時也發生錯誤: {save_e}")


# --- Save Final Model (only if training completed successfully) ---
if training_successful:
    stats_path = f"/kaggle/working/vecnormalize_stats_{run_id}.pkl"
    final_model_name = f'{STUDENT_ID}_dqn_final_{run_id}.zip'
    final_model_path = os.path.join("/kaggle/working", final_model_name)

    try:
        # Save the VecNormalize statistics, which are crucial for loading the model later
        # Ensure the train_env is not None and is indeed VecNormalize
        if train_env and isinstance(train_env, VecNormalize):
            train_env.save(stats_path)
            write_log(f"    VecNormalize 統計數據已儲存至 {stats_path}")
            if wandb_enabled and run: wandb.save(stats_path) # Upload stats to wandb
        else:
             write_log("⚠️ train_env 不是 VecNormalize 實例或為 None, 無法儲存統計數據.")


        model.save(final_model_path)
        write_log(f"✅ 最終模型已儲存: {final_model_path}")
        if 'kaggle_env' in locals() and kaggle_env: # Only display FileLink in Kaggle
             display(FileLink(final_model_path))
        if wandb_enabled and run: wandb.save(final_model_path) # Upload final model to wandb

    except Exception as e:
        write_log(f"❌ 儲存最終模型或統計數據時出錯: {e}", exc_info=True)
        training_successful = False # Mark as unsuccessful if saving fails


# ----------------------------
# Evaluation (only if training and saving were successful)
# ----------------------------
if training_successful:
    write_log("\n🧪 開始評估訓練後的模型...")

    # Create a separate evaluation environment
    eval_env = None # Initialize eval_env to None
    try:
        # Ensure evaluation env is created with render_mode if needed for GIF
        # Use lambda to ensure a new environment is created each time
        eval_env_base = DummyVecEnv([lambda: TetrisEnv(render_mode="rgb_array" if wandb_enabled else None)]) # Set render_mode for base env

        # Wrap with FrameStack FIRST, same as training
        n_stack_eval = run.config.get("n_stack", config["n_stack"]) if run else config["n_stack"]
        eval_env_stacked = VecFrameStack(eval_env_base, n_stack=n_stack_eval, channels_order="first")

        # Load the SAME VecNormalize statistics used during training
        if os.path.exists(stats_path):
             write_log(f"    載入 VecNormalize 統計數據從 {stats_path}")
             eval_env = VecNormalize.load(stats_path, eval_env_stacked)
             eval_env.training = False # Set mode to evaluation
             eval_env.norm_reward = False # IMPORTANT: View actual rewards, not normalized ones
             write_log("    評估環境建立成功並載入 VecNormalize 統計數據.")
        else:
             write_log(f"❌ 錯誤: VecNormalize 統計文件未找到於 {stats_path}。無法載入統計數據進行評估。")
             # If stats not found, evaluation might be skewed, or skip normalization entirely
             # For robustness, let's proceed without normalization if file is missing, but log a warning
             write_log("⚠️ 將嘗試在沒有載入 VecNormalize 統計數據的情況下進行評估 (請注意分數可能不同).")
             eval_env = eval_env_stacked # Use the frame-stacked env directly
             # Ensure attributes expected by the eval loop exist even if not VecNormalize
             # This mock is needed because VecFrameStack's default get_attr might not expose
             # the base environment list in the same way VecNormalize does, or we need
             # specific access patterns.
             # MODIFIED: Correct the lambda to capture eval_env_stacked and use correct path
             # The path from VecFrameStack to TetrisEnv is eval_env_stacked.envs.envs[i]
             write_log("    設置 mock get_attr 方法用於評估環境...")
             # Corrected lambda: captures eval_env_stacked and navigates down wrappers
             # VecFrameStack -> DummyVecEnv (.envs) -> [TetrisEnv] (.envs)
             eval_env.get_attr = lambda attr_name, indices=None, _self_fs_env=eval_env_stacked: [
                 getattr(_self_fs_env.envs.envs[i], attr_name) # Access attribute on the base TetrisEnv
                 for i in (indices if indices is not None else range(len(_self_fs_env.envs.envs))) # Iterate through base envs
             ]
             write_log("    已為 eval_env (VecFrameStack) 設置 mock get_attr 方法.")


    except Exception as e:
        write_log(f"❌ 建立評估環境時出錯: {e}", exc_info=True)
        eval_env = None

    if eval_env is not None:
        # --- Run Evaluation Episodes ---
        num_eval_episodes = 10 # Evaluate for 10 episodes for better average
        total_rewards = []
        total_lines = []
        total_lifetimes = []
        all_frames = [] # For GIF of the first episode

        write_log(f"    執行 {num_eval_episodes} 輪評估...")
        try:
            for i in range(num_eval_episodes):
                write_log(f"    > 開始評估 Episode {i+1}")
                obs, _ = eval_env.reset() # SB3 reset returns (obs, info)
                # Handle potential Tuple obs from reset if using non-normalized env fallback (though SB3 resets usually handle this)
                # if isinstance(obs, tuple):
                #     obs = obs[0] # Take the observation part
                done = False
                episode_reward = 0
                episode_lines = 0
                episode_lifetime = 0
                frames = []
                last_info = {}
                step_count = 0 # Track steps manually for safety

                while not done:
                    # Render base env for GIF (only for first episode if wandb enabled)
                    # Use get_attr to access the base environment's render method
                    if i == 0 and wandb_enabled: # Check if wandb is enabled for logging GIF
                         try:
                             # Access the underlying TetrisEnv instance via get_attr
                             # eval_env (VecNormalize or VecFrameStack) -> .get_attr("envs") -> [VecFrameStack or DummyVecEnv] -> element[0] -> .envs -> [TetrisEnv] -> element[0]
                             # Let's use get_attr to get the list of base envs directly if possible, or navigate.
                             # The mock get_attr returns a list where each element is a base env attribute.
                             # If we call get_attr("."), it might return the base env itself? Let's try getting the 'env' attribute from the level above the base env.
                             # Access the first element from the underlying DummyVecEnv's envs list
                             # eval_env -> (VecNormalize/VecFrameStack) -> .envs[0] -> (DummyVecEnv) -> .envs[0] -> TetrisEnv
                             # Use eval_env.get_attr('envs') to get list of wrapped envs (DummyVecEnv in this case)
                             # Then access the first DummyVecEnv and its list of unwrapped envs
                             wrapped_env_list = eval_env.get_attr('envs')
                             if wrapped_env_list and hasattr(wrapped_env_list[0], 'envs') and wrapped_env_list[0].envs:
                                  base_env_instance = wrapped_env_list[0].envs[0] # Get the actual TetrisEnv instance
                                  if hasattr(base_env_instance, 'render') and base_env_instance.render_mode == "rgb_array": # Ensure base env has render method and mode
                                       raw_frame = base_env_instance.render(mode="rgb_array")
                                  else:
                                      raw_frame = None # Cannot render if mode is not set or no render method
                                      if i == 0 and not hasattr(self, '_eval_render_mode_error_reported'):
                                          write_log("⚠️ 評估時基礎環境的 render_mode 未設置為 'rgb_array'，或無 render 方法，無法收集 GIF 幀.")
                                          self._eval_render_mode_error_reported = True

                             else:
                                raw_frame = None
                                if i == 0 and not hasattr(self, '_eval_render_error_reported_access'):
                                     write_log("⚠️ 評估時無法通過 get_attr 或直接屬性訪問到底層 TetrisEnv 實例來獲取渲染幀.")
                                     self._eval_render_error_reported_access = True


                             if raw_frame is not None:
                                 # Resize the 84x84 RGB array to a larger size for the GIF
                                 gif_frame = cv2.resize(raw_frame, (256, 256), interpolation=cv2.INTER_NEAREST) # Scale resized frame
                                 frames.append(gif_frame)
                         except Exception as render_err:
                             # Log render error once per evaluation run
                             if not hasattr(self, '_eval_render_error_reported') or not self._eval_render_error_reported:
                                 write_log(f"⚠️ 評估時獲取渲染幀出錯: {render_err}", exc_info=True)
                                 self._eval_render_error_reported = True # Prevent spam


                    # Predict and step using the trained model
                    action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for evaluation

                    # Step the evaluation environment
                    # SB3 step returns (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, infos = eval_env.step(action)

                    # ensure infos is a list of dicts (VecEnv standard)
                    if not isinstance(infos, list):
                         # Handle unexpected info format, try to use it directly if possible
                         last_info = infos
                         # write_log("⚠️ eval_env.step returned infos not as a list. Check environment wrapper.") # Too noisy
                    elif infos: # List is not empty
                         last_info = infos[0] # Get info from the first (and only) environment
                    else: # infos is an empty list
                         last_info = {} # Default to empty dict

                    # Accumulate rewards and stats (remember reward is NOT normalized here if norm_reward=False)
                    # SB3 VecEnv step returns rewards as a numpy array
                    if isinstance(reward, np.ndarray):
                         episode_reward += reward[0]
                    else:
                         episode_reward += reward # Handle scalar reward (shouldn't happen with VecEnv)
                         # write_log("⚠️ eval_env.step returned scalar reward. Check environment wrapper.") # Too noisy

                    # Use .get() for safety, default to previous value if key missing
                    # Ensure correct keys from TetrisEnv's info dict
                    # These keys are set in TetrisEnv.step -> info dict
                    episode_lines = last_info.get('removed_lines', episode_lines)
                    episode_lifetime = last_info.get('lifetime', episode_lifetime) # Use lifetime from info if available
                    step_count += 1 # Fallback step counter

                    # Check for termination from either 'terminated' or 'truncated' flags
                    done = terminated or truncated

                    # Add a safety break for evaluation episodes to prevent infinite loops
                    if step_count > 3000: # Increased limit again for potentially longer games
                        write_log(f"⚠️ 評估 Episode {i+1} 超過 {step_count} 步, 強制終止.")
                        done = True # Force end the episode
                        # Add truncated flag if ended by step limit
                        if not terminated: truncated = True


                # Episode finished
                # Use the lifetime from the last info received if available, otherwise the step count
                final_episode_lifetime = last_info.get('lifetime', step_count)

                write_log(f"    < 完成評估 Episode {i+1}: Reward={episode_reward:.2f}, Lines={episode_lines}, Steps={final_episode_lifetime}")
                total_rewards.append(episode_reward)
                total_lines.append(episode_lines)
                total_lifetimes.append(final_episode_lifetime) # Use the final lifetime

                if i == 0: all_frames = frames # Store frames from the first episode

            write_log(f"--- 評估結果 ({num_eval_episodes} episodes) ---")
            # Calculate and print aggregate statistics
            if total_rewards: # Ensure lists are not empty
                 mean_reward = np.mean(total_rewards)
                 std_reward = np.std(total_rewards)
                 mean_lines = np.mean(total_lines)
                 std_lines = np.std(total_lines)
                 mean_lifetime = np.mean(total_lifetimes)
                 std_lifetime = np.std(total_lifetimes)

                 write_log(f"    平均 Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                 write_log(f"    平均 Lines: {mean_lines:.2f} +/- {std_lines:.2f}")
                 write_log(f"    平均 Steps: {mean_lifetime:.2f} +/- {std_lifetime:.2f}")

                 # Log evaluation metrics to Wandb
                 if wandb_enabled and run:
                     wandb.log({
                         "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                         "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                         "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime,
                         "global_step": model.num_timesteps # Log eval results at the final training step
                     })
            else:
                 write_log("    沒有完成任何評估回合.")


            # --- Generate Replay GIF ---
            if all_frames and wandb_enabled: # Only generate GIF if frames were collected and wandb is enabled
                 gif_path = f'/kaggle/working/replay_eval_{run_id}.gif'
                 write_log(f"💾 正在儲存評估回放 GIF 至 {gif_path}...")
                 try:
                     # Ensure frames are uint8 and not None
                     imageio.mimsave(gif_path, [np.array(frame).astype(np.uint8) for frame in all_frames if frame is not None], fps=15, loop=0)
                     write_log("    GIF 儲存成功.")
                     if 'kaggle_env' in locals() and kaggle_env: # Only display FileLink in Kaggle
                          display(FileLink(gif_path))
                     if wandb_enabled and run:
                         wandb.log({"eval/replay": wandb.Video(gif_path, fps=15, format="gif"), "global_step": model.num_timesteps}) # Log GIF to Wandb
                 except Exception as e: write_log(f"    ❌ 儲存 GIF 時發生錯誤: {e}", exc_info=True)
            elif wandb_enabled: # Log why GIF wasn't saved if wandb is enabled
                 write_log("    ⚠️ 未能儲存 GIF (沒有收集到幀, 第一輪評估出錯, 或 Wandb 未啟用).")


            # --- Save Evaluation Results CSV ---
            if total_lines: # Only save CSV if there's data
                 csv_filename = f'tetris_evaluation_scores_{run_id}.csv'
                 csv_path = os.path.join("/kaggle/working", csv_filename)
                 write_log(f"💾 正在儲存評估分數 CSV 至 {csv_path}...")
                 try:
                     with open(csv_path, 'w') as fs:
                         fs.write('episode_id,removed_lines,played_steps,reward\n')
                         # Ensure lists are not empty before accessing index
                         for i in range(len(total_lines)):
                              fs.write(f'eval_{i},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                         # Write average row if data exists
                         if total_rewards: # Use total_rewards to check if any episode finished
                             fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                     write_log(f"✅ 評估分數 CSV 已儲存: {csv_path}")
                     if 'kaggle_env' in locals() and kaggle_env: # Only display FileLink in Kaggle
                         display(FileLink(csv_path))
                     if wandb_enabled and run: wandb.save(csv_path) # Upload CSV to wandb
                 except Exception as e: write_log(f"    ❌ 儲存 CSV 時發生錯誤: {e}", exc_info=True)
            else:
                 write_log("    ⚠️ 沒有評估數據可以儲存為 CSV.")

        except Exception as eval_e:
            write_log(f"❌ 評估迴圈中發生錯誤: {eval_e}", exc_info=True)

        finally:
            # Ensure evaluation env is closed even if errors occur
            if eval_env:
                 # Check if eval_env has a close method (VecEnvs do)
                 if hasattr(eval_env, 'close'):
                    eval_env.close()
                    write_log("    評估環境已關閉.")
                 # Also explicitly close the base env if it's different and has a close method
                 # This part might be overly cautious depending on VecEnv implementations,
                 # but safer for custom setups.
                 if 'eval_env_stacked' in locals() and eval_env is not eval_env_stacked and hasattr(eval_env_stacked, 'close'):
                      # Check if eval_env_stacked is not None before closing
                      if eval_env_stacked:
                           eval_env_stacked.close()
                           write_log("    評估堆疊環境已關閉.")
                 if 'eval_env_base' in locals() and eval_env_stacked is not eval_env_base and hasattr(eval_env_base, 'close'):
                      # Check if eval_env_base is not None before closing
                      if eval_env_base:
                           eval_env_base.close()
                           write_log("    評估基礎向量環境已關閉.")


# --- Cleanup ---
write_log("🧹 清理環境...")
# Ensure training env is closed
if 'train_env' in locals() and train_env: # Check if train_env exists and is not None
    try:
        train_env.close()
        write_log("    訓練環境已關閉.")
    except Exception as e:
        write_log(f"    關閉訓練環境時出錯: {e}")

# Close the Java server process
if java_process and java_process.poll() is None: # Check if process exists and is running (poll() is None means running)
      write_log("    正在終止 Java server process...")
      try:
          java_process.terminate() # Send SIGTERM
          # Add a timeout for graceful termination
          java_process.wait(timeout=10) # Wait up to 10 seconds
          write_log("    Java server process 已終止.")
      except subprocess.TimeoutExpired:
          write_log("    Java server 未能在 10 秒內終止, 強制結束...")
          java_process.kill() # Send SIGKILL
          write_log("    Java server process 已強制結束.")
      except Exception as e:
          write_log(f"    終止 Java server process 時發生錯誤: {e}")

elif java_process and java_process.poll() is not None: # Process exists but is not running
    write_log("    Java server process 已自行結束.")
else: # Process object doesn't exist
    write_log("    Java server process 未啟動.")


# Finish the Wandb run if it was initialized and training didn't crash early
# Ensure run is finalized regardless of success, but mark failure if needed
if run: # Check if run object exists
    # Check if run is still running before trying to finish
    # Use run.finish() directly; it handles the state internally
    # Passing exit_code helps signal success (0) or failure (non-zero) in Wandb UI
    exit_code = 0 if training_successful else 1
    # Check if the run is already finished (e.g., by an earlier exception handler)
    if hasattr(run, '_run_state') and run._run_state != 'finished':
         try:
              run.finish(exit_code=exit_code)
              write_log(f"✨ Wandb run finished (exit code {exit_code}).")
         except Exception as e:
              write_log(f"❌ Error finishing Wandb run: {e}")
    elif not hasattr(run, '_run_state'): # Fallback for older wandb versions
         try:
              run.finish(exit_code=exit_code)
              write_log(f"✨ Wandb run finished (exit code {exit_code}, fallback).")
         except Exception as e:
              # print("Wandb run might have been finished already.")
              pass # Assume it was already finished


write_log("🏁 腳本執行完畢.")