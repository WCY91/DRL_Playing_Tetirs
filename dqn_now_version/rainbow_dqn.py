# -*- coding: utf-8 -*-
import numpy as np
import socket
import cv2
import subprocess
import os
import time

from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
# Removed unused imports: shutil, glob, matplotlib.pyplot, IPython.display.FileLink, IPython.display.display
# Removed unused imports: stable_baselines3.common.env_checker, stable_baselines3.common.env_util, stable_baselines3.common.callbacks
import imageio
import gymnasium as gym
from gymnasium import spaces
# Import RainbowDQN from sb3_contrib
# YOU NEED TO INSTALL sb3-contrib: pip install sb3-contrib
try:
    from sb3_contrib import RainbowDQN
    write_log("✅ 成功導入 sb3_contrib.RainbowDQN")
except ImportError:
    write_log("❌ 錯誤: 未找到 sb3_contrib 套件。請使用 'pip install sb3-contrib' 安裝。")
    # Define a dummy class or exit gracefully if import fails
    class RainbowDQN:
        def __init__(self, *args, **kwargs):
            raise ImportError("sb3_contrib.RainbowDQN is not installed.")
        def learn(self, *args, **kwargs):
             raise ImportError("sb3_contrib.RainbowDQN is not installed.")
        def save(self, *args, **kwargs):
             raise ImportError("sb3_contrib.RainbowDQN is not installed.")
        def predict(self, *args, **kwargs):
             raise ImportError("sb3_contrib.RainbowDQN is not installed.")
        @property
        def device(self):
             return "cpu" # Dummy device
        def get_parameters(self):
             return {} # Dummy params

import torch
import time
import pygame
# Removed unused import: PPO
# --- Wandb Setup ---
# os imported above
import wandb
from kaggle_secrets import UserSecretsClient
# Import WandbCallback for SB3 integration
from wandb.integration.sb3 import WandbCallback

# --- Configuration ---
# Set your student ID here for filenames
STUDENT_ID = "113598065"
# Set total training steps
TOTAL_TIMESTEPS = 3000000 # INCREASED for better training
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
# --- Wandb Login and Initialization ---
try:
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login()
    wandb_enabled = True
    write_log("✅ Wandb 登入成功!")
except Exception as e:
    write_log(f"⚠️ Wandb 登入失敗 (可能是沒有設定 Secrets 或 API Key 過期): {e}. 將在沒有 Wandb 記錄的情況下執行。")
    wandb_enabled = False
    WANDB_API_KEY = None # Ensure it's None if not available

# Start a wandb run if enabled
# --- !!! MODIFY HYPERPARAMETERS HERE for Wandb logging if needed !!! ---
# Using parameters specifically for RainbowDQN
config = { # Log hyperparameters
    "policy_type": "CnnPolicy", # RainbowDQN supports CnnPolicy
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1",
    "gamma": 0.99,
    "learning_rate": 1e-4, # Common for DQN/RainbowDQN
    "buffer_size": 500000, # Larger buffer size often helps
    "learning_starts": 5000, # Start learning after sufficient random exploration
    "batch_size": 32, # Common batch size
    # RainbowDQN specific parameters
    "n_step_returns": 5, # Number of steps for multi-step learning (typical values 3-10)
    "noisy_nets": True, # Enable Noisy Nets for exploration
    "v_min": -10.0, # Min value for Distributional RL (adjust based on expected reward range)
    "v_max": 10.0, # Max value for Distributional RL (adjust based on expected reward range)
    "n_quantiles": 51, # Number of quantiles for Distributional RL (typical is 51)
    "target_update_interval": 5000, # Update target network less frequently
    "exploration_fraction": 0.5, # Still relevant if not using NoisyNets, but often reduced or ignored with it
    "exploration_final_eps": 0.01, # Often set very low if using NoisyNets, or 0
    "prioritized_replay": True, # Enable Prioritized Experience Replay
    "prioritized_replay_alpha": 0.6, # Alpha parameter for PER
    "prioritized_replay_beta0": 0.4, # Initial beta for PER (SB3 uses beta0)
    # Removed double_dqn, it's implicit in RainbowDQN

    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Reward Coefficients ---
    "reward_line_clear_coeff": 100.0, # Base reward for clearing 1 line
    "penalty_height_increase_coeff": 5.0, # Penalty for height increase (per row added)
    "penalty_hole_increase_coeff": 10.0, # Penalty for new holes (per hole added)
    "penalty_step_coeff": 0.0,# Penalty for each step (survival penalty - set to 0)
    "penalty_game_over_coeff": 200.0, # Penalty for game over
    "reward_height_decrease_coeff": 15.0 # NEW: Reward for height decrease (per row removed)
}

if wandb_enabled:
    run = wandb.init(
        project="tetris-training-rainbow-dqn", # Project name
        entity="t113598065-ntut-edu-tw", # Replace with your Wandb entity if different
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        config=config # Log hyperparameters from the dictionary
    )
    run_id = run.id # Get run ID for saving paths
    write_log(f"✨ Wandb run initialized: {run.url}")
else:
    run = None # Set run to None if wandb is disabled
    run_id = f"local_{int(time.time())}" # Create a local ID for paths
    write_log("⚠️ Wandb 已停用，訓練將不會記錄到 Wandb。")


log_path = f"/kaggle/working/tetris_train_log_{run_id}.txt"
# Ensure log file exists or is created
try:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- Log started for run {run_id} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
except Exception as e:
    print(f"Error creating log file {log_path}: {e}")





def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60):
    """Waits for the Tetris TCP server to become available."""
    write_log(f"⏳ 等待 Tetris TCP server 啟動中 ({ip}:{port})...")
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((ip, port), timeout=1.0) as test_sock:
                 # Optional: Send a small keep-alive byte to confirm responsiveness
                 test_sock.sendall(b'\x00') # Send dummy byte, server should ignore
                 # Optional: Try receiving a byte if server sends anything on connect/dummy command
                 # try:
                 #    test_sock.recv(1)
                 # except socket.timeout:
                 #    pass # Server didn't send anything, that's fine
            write_log("✅ Java TCP server 準備完成，連線成功")
            return True # Indicate success
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            if time.time() - start_time > timeout:
                write_log(f"❌ 等待 Java TCP server 超時 ({timeout}s). 錯誤: {e}")
                return False # Indicate failure
            # write_log(f"   連線失敗 ({e}), 1秒後重試...") # Too verbose
            time.sleep(1.0) # Wait a bit longer before retrying
        except Exception as e:
             write_log(f"❌ 等待 Java TCP server 時發生未知錯誤: {e}")
             return False # Indicate failure


# --- Start Java Server ---
java_process = None # Initialize to None
try:
    write_log("🚀 嘗試啟動 Java Tetris server...")
    jar_file = "TetrisTCPserver_v0.6.jar"
    if not os.path.exists(jar_file):
        write_log(f"❌ 錯誤: 找不到 JAR 檔案 '{jar_file}'。請確保它在工作目錄中。")
        raise FileNotFoundError(f"JAR file '{jar_file}' not found.")

    # Start process, redirect stdout/stderr to a file or DEVNULL to keep console clean
    # Using DEVNULL for simplicity here
    java_process = subprocess.Popen(
        ["java", "-jar", jar_file],
        stdout=subprocess.DEVNULL, # Optional: hide server stdout
        stderr=subprocess.DEVNULL # Optional: hide server stderr
    )
    write_log(f"✅ Java server process 啟動 (PID: {java_process.pid})")
    if not wait_for_tetris_server():
        raise TimeoutError("Java server did not become available within the timeout.") # Raise specific error

except Exception as e:
    write_log(f"❌ 啟動或等待 Java server 時發生錯誤: {e}")
    # Attempt to terminate if process started but failed connection
    if java_process and java_process.poll() is None:
        write_log("   嘗試終止未成功連接的 Java server process...")
        java_process.terminate()
        try:
            java_process.wait(timeout=5) # Wait a bit longer
        except subprocess.TimeoutExpired:
            write_log("   Java server 未能在時限內終止，強制結束...")
            java_process.kill()
    if run: run.finish(exit_code=1, quiet=True) # Ensure wandb run is marked as failed
    raise # Re-raise the exception to stop the script


# --- Check GPU ---
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    write_log(f"✅ PyTorch is using GPU: {device_name}")
else:
    write_log("⚠️ PyTorch is using CPU. Training will be significantly slower.")
    write_log("   請考慮在 Kaggle Notebook 設定中啟用 GPU。")

# ----------------------------
# 定義 Tetris 環境 (採用老師的格式, 結合獎勵機制概念)
# ----------------------------
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris that interacts with a Java TCP server."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS = 5
    IMG_HEIGHT = 200
    IMG_WIDTH = 100
    IMG_CHANNELS = 3
    RESIZED_DIM = 84

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        # SB3 VecFrameStack expects channel first (N, C, H, W)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), # (Channels, Height, Width)
            dtype=np.uint8
        )
        self.server_ip = host_ip
        self.server_port = host_port
        self.client_sock = None
        self._connect_socket() # Connect in init

        # Reward shaping & statistics variables
        self.lines_removed = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None # Store original size frame for rendering if needed

        # --- Reward Shaping Coefficients ---
        # Retrieve from Wandb config if available, otherwise use defaults
        current_config = run.config if run else config # Use global config if no run
        self.reward_line_clear_coeff = current_config.get("reward_line_clear_coeff", 100.0)
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 5.0)
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 10.0)
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.0)
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_coeff", 200.0)
        # Densification Reward Coefficient
        self.reward_height_decrease_coeff = current_config.get("reward_height_decrease_coeff", 15.0)

        write_log(f"TetrisEnv initialized with Reward Coeffs: Line={self.reward_line_clear_coeff}, H+={self.penalty_height_increase_coeff}, O+={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}, H-={self.reward_height_decrease_coeff}")


        # For rendering
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False # Track Pygame init state
        # Flag to prevent Wandb log error spam
        self._wandb_log_error_reported = False

    def _initialize_pygame(self):
        """Initializes Pygame if not already done."""
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                import pygame
                if not pygame.get_init():
                    pygame.init()
                if not pygame.display.get_init():
                    pygame.display.init()

                # Calculate scale factor to make window larger
                scale_factor = 4

                # Determine render frame source dimensions for scaling
                # If last_raw_render_frame exists, use its shape, otherwise default
                if hasattr(self, 'last_raw_render_frame') and self.last_raw_render_frame is not None:
                    render_height, render_width = self.last_raw_render_frame.shape[:2]
                else:
                    render_height, render_width = self.IMG_HEIGHT, self.IMG_WIDTH # Use original server dimensions

                # Adjust window size based on original aspect ratio if needed,
                # but resizing to source_dim * scale_factor is often simpler for display
                window_width = render_width * scale_factor
                window_height = render_height * scale_factor


                self.window_surface = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log("   Pygame initialized for rendering ('human' mode).")
            except ImportError:
                write_log("⚠️ Pygame not installed, cannot use 'human' render mode.")
                self.render_mode = None # Disable human rendering
            except Exception as e:
                write_log(f"⚠️ Error initializing Pygame: {e}")
                self.render_mode = None

    def _connect_socket(self):
        """Establishes connection to the game server."""
        attempts = 3
        for i in range(attempts):
            try:
                if self.client_sock:
                    self.client_sock.close()
                self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_sock.settimeout(10.0)
                self.client_sock.connect((self.server_ip, self.server_port))
                # write_log(f"🔌 Socket connected to {self.server_ip}:{self.server_port}") # Less verbose
                return # Success
            except socket.error as e:
                 if i < attempts - 1:
                     write_log(f"❌ Socket connection error during connect (Attempt {i+1}/{attempts}): {e}. Retrying...")
                     time.sleep(1.0) # Wait before retrying
                 else:
                    write_log(f"❌ Socket connection error during connect (Final Attempt): {e}")
                    raise ConnectionError(f"Failed to connect to Tetris server at {self.server_ip}:{self.server_port} after {attempts} attempts.")

    def _send_command(self, command: bytes):
        """Sends a command to the server, handles potential errors."""
        if not self.client_sock:
            # Attempt to reconnect if socket is somehow lost
            write_log("⚠️ Socket not connected when trying to send. Attempting to reconnect...")
            try:
                self._connect_socket()
                write_log("✅ Reconnected successfully.")
            except ConnectionError as e:
                write_log(f"❌ Reconnection failed: {e}")
                raise ConnectionAbortedError("Socket is not connected and reconnection failed.")

        try:
            self.client_sock.sendall(command)
        except socket.timeout:
            write_log("❌ Socket timeout during send.")
            raise ConnectionAbortedError("Socket timeout during send")
        except socket.error as e:
            write_log(f"❌ Socket error during send: {e}")
            raise ConnectionAbortedError(f"Socket error during send: {e}")

    def _receive_data(self, size):
        """Receives exactly size bytes from the server."""
        if not self.client_sock:
            # Attempt to reconnect if socket is somehow lost
            write_log("⚠️ Socket not connected when trying to receive. Attempting to reconnect...")
            try:
                self._connect_socket()
                write_log("✅ Reconnected successfully.")
            except ConnectionError as e:
                write_log(f"❌ Reconnection failed: {e}")
                raise ConnectionAbortedError("Socket is not connected and reconnection failed.")

        data = b""
        try:
            self.client_sock.settimeout(10.0) # Set timeout for recv
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    write_log(f"❌ Socket connection broken during receive (expected {size}, received {len(data)}). Received empty chunk.")
                    raise ConnectionAbortedError("Socket connection broken")
                data += chunk
        except socket.timeout:
             write_log(f"❌ Socket timeout during receive (expected {size}, got {len(data)}).")
             raise ConnectionAbortedError("Socket timeout during receive")
        except socket.error as e:
            write_log(f"❌ Socket error during receive: {e}")
            raise ConnectionAbortedError(f"Socket error during receive: {e}")
        return data

    def get_tetris_server_response(self):
        """Gets state update from the Tetris server via socket."""
        try:
            is_game_over_byte = self._receive_data(1)
            is_game_over = (is_game_over_byte == b'\x01')

            removed_lines_bytes = self._receive_data(4)
            removed_lines = int.from_bytes(removed_lines_bytes, 'big')

            height_bytes = self._receive_data(4)
            height = int.from_bytes(height_bytes, 'big')

            holes_bytes = self._receive_data(4)
            holes = int.from_bytes(holes_bytes, 'big')

            img_size_bytes = self._receive_data(4)
            img_size = int.from_bytes(img_size_bytes, 'big')

            if img_size <= 0 or img_size > 2000000: # Increased max size just in case
                 write_log(f"❌ Received invalid image size: {img_size}. Aborting receive.")
                 raise ValueError(f"Invalid image size received: {img_size}")

            img_png = self._receive_data(img_size)

            # Decode and preprocess image
            nparr = np.frombuffer(img_png, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if np_image is None:
                 write_log("❌ Failed to decode image from server response.")
                 # Return last known state and signal termination
                 return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

            # Store original size frame for rendering if needed
            self.last_raw_render_frame = np_image.copy()

            # Resize and convert to grayscale, add channel dimension for SB3
            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8) # (1, H, W)

            # Store processed obs
            self.last_observation = observation.copy()

            return is_game_over, removed_lines, height, holes, observation

        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e:
             write_log(f"❌ Connection/Value error getting server response: {e}. Ending episode.")
             # Return last known state and signal termination
             return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"❌ Unexpected error getting server response: {e}. Ending episode.", exc_info=True)
            # Return last known state and signal termination
            return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()


    def step(self, action):
        # --- Send Action ---
        command_map = {
            0: b"move -1\n", 1: b"move 1\n",
            2: b"rotate 0\n", 3: b"rotate 1\n",
            4: b"drop\n"
        }
        command = command_map.get(action)
        if command is None:
            write_log(f"⚠️ Invalid action received: {action}. Sending 'drop'.")
            command = b"drop\n"

        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            write_log(f"❌ Ending episode due to send failure in step: {e}")
            terminated = True
            observation = self.last_observation.copy() # Use last good observation
            reward = self.penalty_game_over_coeff * -1 # Apply game over penalty directly
            info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'send_error'}
            info['terminal_observation'] = observation # Add terminal observation

            # --- Log detailed rewards on send failure termination ---
            if wandb_enabled and run:
                try:
                    log_data = {
                         "reward/step_total": reward,
                         "reward/step_line_clear": 0.0,
                         "reward/step_height_penalty": 0.0,
                         "reward/step_hole_penalty": 0.0,
                         "reward/step_survival_penalty": 0.0,
                         "reward/step_height_decrease": 0.0, # Log 0 for this
                         "reward/step_game_over_penalty": -self.penalty_game_over_coeff, # Log the penalty
                         "env/lines_cleared_this_step": 0,
                         "env/height_increase": 0,
                         "env/hole_increase": 0,
                         "env/current_height": self.current_height,
                         "env/current_holes": self.current_holes,
                         "env/current_lifetime": self.lifetime
                    }
                    # Filter out zero reward components (except game over) for cleaner graphs
                    filtered_log_data = {k: v for k, v in log_data.items() if not (k.startswith("reward/") and not k.endswith("game_over_penalty") and v == 0) or k.startswith("env/")}
                    wandb.log(filtered_log_data) # Log immediately
                except Exception as log_e:
                     if not self._wandb_log_error_reported:
                         write_log(f"Wandb logging error in step (send fail): {log_e}", exc_info=True)
                         self._wandb_log_error_reported = True
            # --- End logging ---

            return observation, reward, terminated, False, info # Return immediately

        # --- Get State Update ---
        terminated, new_lines_removed, new_height, new_holes, observation = self.get_tetris_server_response()

        # --- Calculate Reward ---
        reward = 0.0
        line_clear_reward = 0.0
        height_penalty = 0.0
        hole_penalty = 0.0
        step_penalty = 0.0 # Default
        game_over_penalty = 0.0 # Default
        height_decrease_reward = 0.0 # NEW: Default

        lines_cleared_this_step = new_lines_removed - self.lines_removed

        # Multi-line clear reward logic
        if lines_cleared_this_step > 0:
            if lines_cleared_this_step == 1:
                line_clear_reward = 1 * self.reward_line_clear_coeff
            elif lines_cleared_this_step == 2:
                line_clear_reward = 3 * self.reward_line_clear_coeff # Example: 3x base for double
            elif lines_cleared_this_step == 3:
                line_clear_reward = 5 * self.reward_line_clear_coeff # Example: 5x base for triple
            elif lines_cleared_this_step >= 4:
                line_clear_reward = 8 * self.reward_line_clear_coeff # Example: 8x base for Tetris
            reward += line_clear_reward

        height_increase = new_height - self.current_height
        if height_increase > 0:
            height_penalty = height_increase * self.penalty_height_increase_coeff
            reward -= height_penalty
        # Reward for Height Decrease
        elif height_increase < 0:
            height_decrease_reward = (-height_increase) * self.reward_height_decrease_coeff
            reward += height_decrease_reward

        hole_increase = new_holes - self.current_holes
        if hole_increase > 0:
            hole_penalty = hole_increase * self.penalty_hole_increase_coeff
            reward -= hole_penalty

        # Step penalty (survival penalty)
        step_penalty = self.penalty_step_coeff
        reward -= step_penalty

        # Game Over Penalty
        if terminated:
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            # Log only once per game over for clarity, ADDED reward breakdown
            write_log(f"💔 Game Over! Final Lines: {new_lines_removed}, Lifetime: {self.lifetime + 1}. Step Reward Breakdown: LC={line_clear_reward:.2f}, H+={-height_penalty:.2f}, H-={height_decrease_reward:.2f}, O+={-hole_penalty:.2f}, SP={-step_penalty:.2f}, GO={-game_over_penalty:.2f} -> Total={reward:.2f}")

        # --- Update Internal State ---
        self.lines_removed = new_lines_removed
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime}
        truncated = False # Typically false for game over, true for time limit etc.

        if terminated:
            info['terminal_observation'] = observation.copy()
            # Log final stats here if needed, or use SB3 logger/callback
            # Example: print(f"Episode End: Lines={self.lines_removed}, Lifetime={self.lifetime}, Reward={reward}")


        # --- Detailed Wandb Logging ---
        if wandb_enabled and run:
             try:
                 log_data = {
                     "reward/step_total": reward,
                     "reward/step_line_clear": line_clear_reward,
                     "reward/step_height_penalty": -height_penalty, # Log penalties as negative values
                     "reward/step_height_decrease": height_decrease_reward, # Log positive reward
                     "reward/step_hole_penalty": -hole_penalty,
                     "reward/step_survival_penalty": -step_penalty,
                     "reward/step_game_over_penalty": -game_over_penalty, # Will be non-zero only on last step
                     "env/lines_cleared_this_step": lines_cleared_this_step,
                     "env/height_increase": height_increase if height_increase > 0 else 0, # Log only positive increase
                     "env/height_decrease": -height_increase if height_increase < 0 else 0, # Log only positive decrease
                     "env/hole_increase": hole_increase if hole_increase > 0 else 0, # Log only positive increase
                     "env/current_height": self.current_height,
                     "env/current_holes": self.current_holes,
                     "env/current_lifetime": self.lifetime # Log lifetime at each step
                 }
                 # Filter out zero reward components (except game over penalty if it's the last step)
                 # and ensure env metrics are always logged
                 filtered_log_data = {k: v for k, v in log_data.items() if (k.startswith("env/") or v != 0.0 or (k == "reward/step_game_over_penalty" and terminated))}

                 # We don't have easy access to the global step here, rely on Wandb/SB3 sync
                 # SB3's WandbCallback handles global step logging
                 wandb.log(filtered_log_data)
             except Exception as log_e:
                 # Prevent spamming logs if Wandb logging fails repeatedly
                 if not self._wandb_log_error_reported:
                     write_log(f"Wandb logging error in step: {log_e}", exc_info=True)
                     self._wandb_log_error_reported = True


        # Optional: Render on step if requested
        if self.render_mode == "human":
             self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the Wandb error reported flag for the new episode
        self._wandb_log_error_reported = False

        for attempt in range(5): # Allow more attempts to reset/reconnect
            try:
                # Ensure connection is fresh before sending start
                self._connect_socket() # Reconnect on reset
                self._send_command(b"start\n")
                # The server sends the initial state immediately after 'start'
                terminated, lines, height, holes, observation = self.get_tetris_server_response()
                if terminated:
                     write_log(f"⚠️ Server reported game over on reset attempt {attempt+1}. Retrying...")
                     if attempt == 4: # Last attempt failed
                          write_log("❌ Server still terminated after multiple reset attempts. Cannot proceed.")
                          raise RuntimeError("Tetris server failed to reset properly after multiple attempts.")
                     time.sleep(0.5) # Small delay before retry
                     continue # Retry the loop
                # Reset successful
                self.lines_removed = 0
                self.current_height = height
                self.current_holes = holes
                self.lifetime = 0
                self.last_observation = observation.copy()
                # write_log(f"🔄 Environment Reset. Initial state: H={height}, O={holes}") # Less verbose logging
                info = {}
                return observation, info

            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError, ValueError) as e:
                 write_log(f"🔌 Connection/Error issue during reset attempt {attempt+1} ({e}). Retrying...")
                 if attempt == 4: # Last attempt failed
                     raise RuntimeError(f"Failed to connect, communicate or reset Tetris server after multiple attempts: {e}")
                 time.sleep(1.0) # Wait a bit longer before retrying


        # Should not be reached if logic is correct, but as fallback:
        raise RuntimeError("Failed to reset Tetris server after all attempts.")


    def render(self):
        self._initialize_pygame() # Ensure pygame is ready if in human mode

        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None:
                 write_log("⚠️ Render called but Pygame window is not initialized.")
                 return

            if hasattr(self, 'last_raw_render_frame') and self.last_raw_render_frame is not None:
                try:
                    # last_raw_render_frame is (H, W, C) BGR from OpenCV
                    render_frame_rgb = cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
                    render_height, render_width, _ = render_frame_rgb.shape

                    # Create surface from the raw frame size
                    surf = pygame.Surface((render_width, render_height))
                    # Blit array requires (W, H, C)
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))

                    # Scale up to the window size
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0))

                    pygame.event.pump() # Process internal Pygame events
                    pygame.display.flip() # Update the full screen surface
                    if self.clock:
                        self.clock.tick(self.metadata["render_fps"]) # Control frame rate
                except Exception as e:
                    write_log(f"⚠️ Error during Pygame rendering: {e}")
                    # Consider setting render_mode to None here if errors persist
            else:
                # Draw a black screen if no frame available yet
                self.window_surface.fill((0, 0, 0))
                pygame.display.flip()

        elif self.render_mode == "rgb_array":
             if hasattr(self, 'last_raw_render_frame') and self.last_raw_render_frame is not None:
                 # Return RGB (H, W, C) from the original size frame if available
                 return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
             else:
                 # Return black frame (original size) if no observation yet
                 return np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8)

    def close(self):
        write_log("🧹 關閉環境連接...")
        if self.client_sock:
            try:
                # Optional: Send a quit command to the server
                # self._send_command(b"quit\n")
                self.client_sock.close()
                write_log("   Socket 已關閉.")
            except socket.error as e:
                write_log(f"   關閉 Socket 時出錯: {e}")
            except Exception as e:
                 write_log(f"   關閉 Socket 時發生未知錯誤: {e}")
            self.client_sock = None

        if self.is_pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.is_pygame_initialized = False
                write_log("   Pygame 已關閉.")
            except Exception as e:
                 write_log(f"   關閉 Pygame 時出錯: {e}")
        write_log("✅ 環境已關閉.")

# --- Environment Setup ---
write_log("✅ 建立基礎環境函數 make_env...")
def make_env():
    """Helper function to create an instance of the Tetris environment."""
    # You can pass render_mode="human" here to see the game while training
    env = TetrisEnv(render_mode=None) # Set to "human" to render during training/eval
    return env

write_log("✅ 建立向量化環境 (DummyVecEnv)...")
# Use DummyVecEnv for single environment interaction
train_env_base = DummyVecEnv([make_env])

write_log("✅ 包裝環境 (VecFrameStack)...")
# Wrap with VecFrameStack (channel-first is important)
# Use wandb config if available, otherwise use default from global config
n_stack = run.config.get("n_stack", config["n_stack"]) if run else config["n_stack"]
train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack, channels_order="first")

write_log("✅ 包裝環境 (VecNormalize - Rewards Only)...")
# Wrap with VecNormalize, NORMALIZING REWARDS ONLY.
# Use wandb config if available, otherwise use default from global config
gamma_param = run.config.get("gamma", config["gamma"]) if run else config["gamma"]
train_env = VecNormalize(train_env_stacked, norm_obs=False, norm_reward=True, gamma=gamma_param)

write_log("   環境建立完成並已包裝 (DummyVecEnv -> VecFrameStack -> VecNormalize)")


# ----------------------------
# RainbowDQN Model Setup and Training
# ----------------------------
write_log("🧠 設定 RainbowDQN 模型...")
# Use wandb config for hyperparameters if available, otherwise use defaults from global config dict
current_config = run.config if run else config # Use global config if no run active

policy_type = current_config.get("policy_type", "CnnPolicy")
learning_rate = current_config.get("learning_rate", 1e-4)
buffer_size = current_config.get("buffer_size", 100000)
learning_starts = current_config.get("learning_starts", 10000)
batch_size = current_config.get("batch_size", 32)

# --- Get RainbowDQN Specific Parameters ---
n_step_returns = current_config.get("n_step_returns", 1) # Default to 1 if not in config
noisy_nets = current_config.get("noisy_nets", False) # Default to False if not in config
v_min = current_config.get("v_min", -10.0) # Default if not in config
v_max = current_config.get("v_max", 10.0) # Default if not in config
n_quantiles = current_config.get("n_quantiles", 51) # Default if not in config
target_update_interval = current_config.get("target_update_interval", 10000)

prioritized_replay = current_config.get("prioritized_replay", False)
prioritized_replay_alpha = current_config.get("prioritized_replay_alpha", 0.6)
prioritized_replay_beta0 = current_config.get("prioritized_replay_beta0", 0.4)


write_log(f"   RainbowDQN Features: N-Step={n_step_returns}, Noisy Nets={noisy_nets}, Distributional RL (v_min={v_min}, v_max={v_max}, quantiles={n_quantiles}), Prioritized Replay={prioritized_replay}")


# Define RainbowDQN model
model = RainbowDQN(
    policy=policy_type,
    env=train_env,
    verbose=1,
    gamma=gamma_param,
    learning_rate=learning_rate,
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    # RainbowDQN specific parameters
    n_step_returns=n_step_returns,
    noisy_nets=noisy_nets,
    v_min=v_min,
    v_max=v_max,
    n_quantiles=n_quantiles,
    # PER parameters (also used by base DQN, but listed here for completeness)
    prioritized_replay=prioritized_replay,
    prioritized_replay_alpha=prioritized_replay_alpha,
    prioritized_replay_beta0=prioritized_replay_beta0,
    # Note: double_dqn is NOT a parameter for RainbowDQN as it's always used

    # Other common parameters
    tau=1.0, # Target network update rate (1.0 for hard update)
    train_freq=(1, "step"), # Train every step
    gradient_steps=1, # Number of gradient steps per training iteration
    target_update_interval=target_update_interval,
    # Exploration parameters - less relevant if noisy_nets=True, but still exist
    exploration_fraction=current_config.get("exploration_fraction", 0.5),
    exploration_final_eps=current_config.get("exploration_final_eps", 0.01),

    policy_kwargs=dict(normalize_images=False), # Keep image normalization off before policy
    seed=42, # Set seed for reproducibility
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log=f"/kaggle/working/runs/{run_id}" if wandb_enabled else None # Log TB only if wandb enabled
)
write_log(f"   模型建立完成. Device: {model.device}")
# write_log(f"   使用的超參數: {model.get_parameters()['policy']}") # Log actual policy params used
write_log(f"   使用的模型超參數: gamma={model.gamma}, lr={model.lr}, buffer_size={model.buffer_size}, learning_starts={model.learning_starts}, batch_size={model.batch_size}, n_step_returns={model.n_step_returns}, noisy_nets={model.noisy_nets}, v_min={model.v_min}, v_max={model.v_max}, n_quantiles={model.n_quantiles}, prioritized_replay={model.prioritized_replay}, prioritized_replay_alpha={model.prioritized_replay_alpha}, prioritized_replay_beta0={model.prioritized_replay_beta0}, target_update_interval={model.target_update_interval}, exploration_fraction={model.exploration_fraction}, exploration_final_eps={model.exploration_final_eps}")


# Setup Wandb callback if enabled
if wandb_enabled:
    # model_save_path includes run_id for unique paths
    model_save_dir = f"/kaggle/working/models/{run_id}"
    os.makedirs(model_save_dir, exist_ok=True) # Ensure directory exists
    wandb_callback = WandbCallback(
        gradient_save_freq=10000, # Log grads every 10k steps
        model_save_path=model_save_dir, # Save models periodically
        model_save_freq=100000, # Save every 100k steps (Increased frequency)
        log="all", # Log histograms, gradients, etc.
        verbose=2
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
        log_interval=10 # Log basic stats (like FPS, mean reward) every 10 episodes to console/TB
    )
    write_log("✅ 訓練完成!")
    training_successful = True
except Exception as e:
     write_log(f"❌ 訓練過程中發生錯誤: {e}", exc_info=True) # Log exception info
     # Save model before exiting if error occurs mid-training
     error_save_path = f'/kaggle/working/{STUDENT_ID}_rainbowdqn_error_save_{run_id}.zip' # Include run_id and updated name
     try:
         model.save(error_save_path)
         write_log(f"   模型已嘗試儲存至 {error_save_path}")
         if wandb_enabled and run: wandb.save(error_save_path) # Upload error model to wandb
     except Exception as save_e:
          write_log(f"   ❌ 儲存錯誤模型時也發生錯誤: {save_e}")
     if run: run.finish(exit_code=1, quiet=True) # Ensure wandb run is marked as failed

# --- Save Final Model (only if training completed successfully) ---
if training_successful:
    stats_path = f"/kaggle/working/vecnormalize_stats_{run_id}.pkl"
    final_model_name = f'{STUDENT_ID}_rainbowdqn_final_{run_id}.zip' # Updated name
    final_model_path = os.path.join("/kaggle/working", final_model_name)

    try:
        train_env.save(stats_path)
        write_log(f"   VecNormalize 統計數據已儲存至 {stats_path}")
        if wandb_enabled and run: wandb.save(stats_path) # Upload stats to wandb

        model.save(final_model_path)
        write_log(f"✅ 最終模型已儲存: {final_model_path}")
        # display(FileLink(final_model_path)) # Removed for cleaner output in console
        if wandb_enabled and run: wandb.save(final_model_path) # Upload final model to wandb

    except Exception as e:
        write_log(f"❌ 儲存最終模型或統計數據時出錯: {e}")
        training_successful = False # Mark as unsuccessful if saving fails


# ----------------------------
# Evaluation (only if training and saving were successful)
# ----------------------------
if training_successful:
    write_log("\n🧪 開始評估訓練後的模型...")

    # Create a separate evaluation environment
    eval_env = None # Initialize eval_env to None
    try:
        # make_env needs to be defined before this block
        eval_env_base = DummyVecEnv([lambda: make_env()]) # Use lambda to ensure fresh env instance

        # Wrap with FrameStack FIRST, same as training
        # Use wandb config if available, otherwise use default from global config
        n_stack_eval = run.config.get("n_stack", config["n_stack"]) if run else config["n_stack"]
        eval_env_stacked = VecFrameStack(eval_env_base, n_stack=n_stack_eval, channels_order="first")

        # Load the SAME VecNormalize statistics
        eval_env = VecNormalize.load(stats_path, eval_env_stacked)
        eval_env.training = False # Set mode to evaluation
        eval_env.norm_reward = False # IMPORTANT: 看 আসল reward (View actual rewards)

        write_log("   評估環境建立成功.")

    except FileNotFoundError:
        write_log(f"❌ 錯誤: VecNormalize 統計文件未找到於 {stats_path}。跳過評估。")
        eval_env = None
    except Exception as e:
        write_log(f"❌ 建立評估環境時出錯: {e}", exc_info=True)
        eval_env = None

    if eval_env is not None:
        # --- Run Evaluation Episodes ---
        num_eval_episodes = 5 # Evaluate for 5 episodes
        total_rewards = []
        total_lines = []
        total_lifetimes = []
        all_frames = [] # For GIF of the first episode

        try:
            for i in range(num_eval_episodes):
                # Reset the environment, handle potential reset errors
                try:
                    obs, _ = eval_env.reset() # reset returns obs, info for VecEnvs
                except Exception as reset_e:
                    write_log(f"❌ 評估 Episode {i+1} 重置環境時出錯: {reset_e}. 跳過此輪評估.")
                    continue # Skip this evaluation episode

                done = False
                episode_reward = 0
                episode_lines = 0
                episode_lifetime = 0
                frames = []
                last_info = {}

                while not done:
                    # Render base env for GIF (only for first episode)
                    if i == 0:
                        try:
                            # Access the underlying TetrisEnv instance
                            # VecNormalize wraps VecFrameStack wraps DummyVecEnv wraps TetrisEnv
                            # Need to get past the layers
                            current_env = eval_env
                            while hasattr(current_env, 'env') or hasattr(current_env, 'envs'):
                                if hasattr(current_env, 'envs'): # For DummyVecEnv
                                    current_env = current_env.envs[0]
                                elif hasattr(current_env, 'env'): # For VecNormalize, VecFrameStack
                                     current_env = current_env.env

                            # Now current_env should be the base TetrisEnv
                            if isinstance(current_env, TetrisEnv):
                                raw_frame = current_env.render(mode="rgb_array")
                                if raw_frame is not None:
                                    frames.append(raw_frame)
                            else:
                                write_log(f"⚠️ 評估時未能獲取原始 TetrisEnv 實例進行渲染. Got type: {type(current_env)}")

                        except Exception as render_err:
                            write_log(f"⚠️ 評估時獲取渲染幀出錯: {render_err}")

                    # Predict and step using the trained model
                    # VecEnvs return a tuple (action, state) from predict
                    action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for evaluation
                    # VecEnvs step returns (obs, reward, terminated, truncated, info) lists
                    obs, reward, terminated, truncated, infos = eval_env.step(action)

                    # Accumulate rewards and stats (remember reward is not normalized here)
                    episode_reward += reward[0] # VecEnv returns lists
                    last_info = infos[0]
                    # Use .get() for safety, default to previous value if key missing
                    # Ensure correct keys from TetrisEnv's info dict
                    episode_lines = last_info.get('removed_lines', episode_lines)
                    episode_lifetime = last_info.get('lifetime', episode_lifetime)
                    done = terminated[0] or truncated[0] # Check both terminated and truncated

                write_log(f"   評估 Episode {i+1}: Reward={episode_reward:.2f}, Lines={episode_lines}, Steps={episode_lifetime}")
                total_rewards.append(episode_reward)
                total_lines.append(episode_lines)
                total_lifetimes.append(episode_lifetime)
                if i == 0: all_frames = frames # Store frames from the first episode

            write_log(f"--- 評估結果 ({num_eval_episodes} episodes) ---")
            # Calculate stats only if episodes were completed
            if total_rewards:
                 mean_reward = np.mean(total_rewards)
                 std_reward = np.std(total_rewards) if len(total_rewards) > 1 else 0.0
                 mean_lines = np.mean(total_lines)
                 std_lines = np.std(total_lines) if len(total_lines) > 1 else 0.0
                 mean_lifetime = np.mean(total_lifetimes)
                 std_lifetime = np.std(total_lifetimes) if len(total_lifetimes) > 1 else 0.0

                 write_log(f"   平均 Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                 write_log(f"   平均 Lines: {mean_lines:.2f} +/- {std_lines:.2f}")
                 write_log(f"   平均 Steps: {mean_lifetime:.2f} +/- {std_lifetime:.2f}")

                 # Log evaluation metrics to Wandb
                 if wandb_enabled and run:
                     wandb.log({
                         "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                         "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                         "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime,
                     })
            else:
                 write_log("   沒有成功完成的評估 Episode。")


            # --- Generate Replay GIF ---
            if all_frames:
                 gif_path = f'/kaggle/working/replay_eval_{run_id}.gif'
                 write_log(f"💾 正在儲存評估回放 GIF 至 {gif_path}...")
                 try:
                     # Ensure frames are valid images
                     valid_frames = [np.array(frame).astype(np.uint8) for frame in all_frames if frame is not None and frame.ndim == 3 and frame.shape[2] == 3]
                     if valid_frames:
                         imageio.mimsave(gif_path, valid_frames, fps=15, loop=0)
                         write_log("   GIF 儲存成功.")
                         # display(FileLink(gif_path)) # Removed for cleaner output
                         if wandb_enabled and run:
                             # Log GIF to Wandb, ensure path is correct for upload
                             wandb.log({"eval/replay": wandb.Video(gif_path, fps=15, format="gif")})
                     else:
                         write_log("   ⚠️ 沒有有效的幀可以儲存為 GIF。")

                 except Exception as e: write_log(f"   ❌ 儲存 GIF 時發生錯誤: {e}", exc_info=True)
            else: write_log("   ⚠️ 未能儲存 GIF (沒有收集到幀或第一輪評估出錯).")

            # --- Save Evaluation Results CSV ---
            if total_lines: # Only save if there's data
                 csv_filename = f'tetris_evaluation_scores_{run_id}.csv'
                 csv_path = os.path.join("/kaggle/working", csv_filename)
                 try:
                     with open(csv_path, 'w') as fs:
                         fs.write('episode_id,removed_lines,played_steps,reward\n')
                         for i in range(len(total_lines)):
                             fs.write(f'eval_{i},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                         # Add average line if calculated
                         if total_rewards and len(total_rewards) > 0:
                              fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')

                     write_log(f"✅ 評估分數 CSV 已儲存: {csv_path}")
                     # display(FileLink(csv_path)) # Removed for cleaner output
                     if wandb_enabled and run: wandb.save(csv_path) # Upload CSV to wandb
                 except Exception as e: write_log(f"   ❌ 儲存 CSV 時發生錯誤: {e}")
            else:
                 write_log("   ⚠️ 沒有評估數據可以儲存為 CSV。")


        except Exception as eval_e:
            write_log(f"❌ 評估迴圈中發生錯誤: {eval_e}", exc_info=True)

        finally:
             # Ensure evaluation env is closed even if errors occur
             if eval_env:
                 eval_env.close()
                 write_log("   評估環境已關閉.")

# --- Cleanup ---
write_log("🧹 清理環境...")
if 'train_env' in locals() and train_env: # Check if train_env exists and is not None
    try:
        train_env.close()
        write_log("   訓練環境已關閉.")
    except Exception as e:
        write_log(f"   關閉訓練環境時出錯: {e}")

# Close the Java server process
if java_process and java_process.poll() is None: # Check if process exists and is running
     write_log("   正在終止 Java server process...")
     java_process.terminate()
     try:
         java_process.wait(timeout=5) # Wait up to 5 seconds
         write_log("   Java server process 已終止.")
     except subprocess.TimeoutExpired:
         write_log("   Java server 未能在 5 秒內終止, 強制結束...")
         java_process.kill()
         write_log("   Java server process 已強制結束.")
elif java_process and java_process.poll() is not None:
     write_log("   Java server process 已自行結束.")
else:
     write_log("   Java server process 未啟動或已關閉.")


# Finish the Wandb run if it was initialized and training didn't crash early
if run: # Check if run object exists
    # Check run.is_running before finishing again to avoid errors if already finished
    # This check is not strictly needed with the finish(exit_code=1) in error handling,
    # but doesn't hurt.
    # if hasattr(run, 'is_running') and run.is_running():
         run.finish()
         write_log("✨ Wandb run finished.")
    # else: # The run was likely finished in the exception handler
    #    pass # No need to print error message again


write_log("🏁 腳本執行完畢.")