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
from stable_baselines3 import PPO
# --- Wandb Setup ---
import os
import wandb
from kaggle_secrets import UserSecretsClient
# Import WandbCallback for SB3 integration
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
# --- Configuration ---
# Set your student ID here for filenames
STUDENT_ID = "113598065"
# Set total training steps
TOTAL_TIMESTEPS = 550000 # Adjust as needed (e.g., 1M, 2M, 5M)


# --- Wandb Login and Initialization ---
try:
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login()
    wandb_enabled = True
except Exception as e:
    print(f"Wandb login failed (running without secrets?): {e}. Running without Wandb logging.")
    wandb_enabled = False
    WANDB_API_KEY = None # Ensure it's None if not available

# Start a wandb run if enabled
# --- !!! MODIFY HYPERPARAMETERS HERE for Wandb logging if needed !!! ---
config = { # Log hyperparameters
    "policy_type": "CnnPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1",
    "gamma": 0.95,
    "learning_rate": 4e-4,
    "buffer_size": 150000,
    "learning_starts": 10,
    "target_update_interval": 1000,
    "exploration_fraction": 0.05, # <<< INCREASED exploration duration
    "exploration_final_eps": 0.03, # Kept final exploration rate
    "batch_size": 64,
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- NEW: Add reward coeffs to config for tracking ---
    "reward_line_clear_coeff": 250.0, # Example value, match below
    "penalty_height_increase_coeff": 1.5, # Example value, match below
    "penalty_hole_increase_coeff": 1.2, # Example value, match below
    "penalty_step_coeff": 0.5, # Example value, match below
    "penalty_game_over_coeff": 30.0 # Example value, match below
}

if wandb_enabled:
    run = wandb.init(
        project="tetris-training-improved",
        entity="t113598065-ntut-edu-tw", # Replace with your Wandb entity if different
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        config=config # Log hyperparameters from the dictionary
    )
    run_id = run.id # Get run ID for saving paths
else:
    run = None # Set run to None if wandb is disabled
    run_id = f"local_{int(time.time())}" # Create a local ID for paths


log_path = f"/kaggle/working/tetris_train_log_{run_id}.txt"

def write_log(message):
    """Appends a message to the log file and prints it."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Error writing to log file {log_path}: {e}")
    print(log_message)

def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60):
    """Waits for the Tetris TCP server to become available."""
    write_log(f"⏳ 等待 Tetris TCP server 啟動中 ({ip}:{port})...")
    start_time = time.time()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
                test_sock.settimeout(1.0)
                test_sock.connect((ip, port))
            write_log("✅ Java TCP server 準備完成，連線成功")
            return True # Indicate success
        except socket.error as e:
            if time.time() - start_time > timeout:
                write_log(f"❌ 等待 Java TCP server 超時 ({timeout}s)")
                return False # Indicate failure
            time.sleep(1.0) # Wait a bit longer before retrying

# --- Start Java Server ---
java_process = None # Initialize to None
try:
    write_log("🚀 嘗試啟動 Java Tetris server...")
    jar_file = "TetrisTCPserver_v0.6.jar"
    if not os.path.exists(jar_file):
         write_log(f"❌ 錯誤: 找不到 JAR 檔案 '{jar_file}'。請確保它在工作目錄中。")
         raise FileNotFoundError(f"JAR file '{jar_file}' not found.")

    # Start process, redirect stdout/stderr to DEVNULL if desired to keep console clean
    java_process = subprocess.Popen(
        ["java", "-jar", jar_file],
        stdout=subprocess.DEVNULL, # Optional: hide server stdout
        stderr=subprocess.DEVNULL  # Optional: hide server stderr
    )
    write_log(f"✅ Java server process 啟動 (PID: {java_process.pid})")
    if not wait_for_tetris_server():
        raise TimeoutError("Java server did not become available.") # Raise specific error

except Exception as e:
    write_log(f"❌ 啟動或等待 Java server 時發生錯誤: {e}")
    # Attempt to terminate if process started but failed connection
    if java_process and java_process.poll() is None:
         write_log("   嘗試終止未成功連接的 Java server process...")
         java_process.terminate()
         try:
             java_process.wait(timeout=2)
         except subprocess.TimeoutExpired:
             java_process.kill()
    raise # Re-raise the exception to stop the script

# --- Check GPU ---
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    write_log(f"✅ PyTorch is using GPU: {device_name}")
else:
    write_log("⚠️ PyTorch is using CPU. Training will be significantly slower.")
class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A very basic Prioritized Experience Replay (PER) buffer.
    """
    def __init__(self, *args, alpha: float = 0.6, epsilon: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        # initialize priorities to maximum so that early samples get high prob
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
    
    def add(self, obs, next_obs, action, reward, done, infos=None):
        idx = self.pos
        super().add(obs, next_obs, action, reward, done, infos=infos)
        # set new priority to max of existing
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        # compute sampling probabilities
        prios = self.priorities[: self.size] + self.epsilon
        probs = prios ** self.alpha
        probs /= probs.sum()
        # sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        # importance‑sampling weights (you’d anneal beta from 0→1)
        beta = self.replay_buffer_kwargs.get("beta", 0.4)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        # get the batch
        batch = super().sample(batch_size, env)
        return batch._replace(weights=weights, indices=indices)
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
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), # (Channels, Height, Width)
            dtype=np.uint8
        )
        self.server_ip = host_ip
        self.server_port = host_port
        self.client_sock = None
        self._connect_socket() # Connect in init
        self.episode_total_reward = 0.0
        # Reward shaping & statistics variables
        self.lines_removed = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # --- !!! REWARD SHAPING COEFFICIENTS MODIFIED HERE !!! ---
        # Retrieve from Wandb config if available, otherwise use defaults
        current_config = run.config if run else config # Use global config if no run
        self.reward_line_clear_coeff = current_config.get("reward_line_clear_coeff", 500.0)       # INCREASED
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 7.5) # DECREASED
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 12.5)   # DECREASED
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.0)                   # SET TO ZERO
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_coeff", 500.0)     # Kept same for now
        write_log(f"TetrisEnv initialized with Reward Coeffs: Line={self.reward_line_clear_coeff}, H={self.penalty_height_increase_coeff}, O={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}")


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
                pygame.init()
                pygame.display.init()
                # Scale window for better visibility
                window_width = self.RESIZED_DIM * 4
                window_height = self.RESIZED_DIM * 4
                self.window_surface = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log("   Pygame initialized for rendering.")
            except ImportError:
                write_log("⚠️ Pygame not installed, cannot use 'human' render mode.")
                self.render_mode = None # Disable human rendering
            except Exception as e:
                write_log(f"⚠️ Error initializing Pygame: {e}")
                self.render_mode = None


    def _connect_socket(self):
        """Establishes connection to the game server."""
        try:
            if self.client_sock:
                self.client_sock.close()
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0)
            self.client_sock.connect((self.server_ip, self.server_port))
            # write_log(f"🔌 Socket connected to {self.server_ip}:{self.server_port}") # Less verbose
        except socket.error as e:
            write_log(f"❌ Socket connection error during connect: {e}")
            raise ConnectionError(f"Failed to connect to Tetris server at {self.server_ip}:{self.server_port}")

    def _send_command(self, command: bytes):
        """Sends a command to the server, handles potential errors."""
        if not self.client_sock:
             raise ConnectionError("Socket is not connected. Cannot send command.")
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
             raise ConnectionError("Socket is not connected. Cannot receive data.")
        data = b""
        try:
            self.client_sock.settimeout(10.0) # Set timeout for recv
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    write_log("❌ Socket connection broken during receive (received empty chunk).")
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

            if img_size <= 0 or img_size > 1000000: # Increased max size slightly
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

            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8) # Combine steps

            # Store frames for rendering/observation
            self.last_raw_render_frame = resized.copy() # Store BGR for render
            self.last_observation = observation.copy() # Store processed obs

            return is_game_over, removed_lines, height, holes, observation

        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e:
             write_log(f"❌ Connection/Value error getting server response: {e}. Ending episode.")
             # Return last known state and signal termination
             return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"❌ Unexpected error getting server response: {e}. Ending episode.")
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
        # write_log(f"{self._log_prefix} Step {self.lifetime + 1}: Chosen Action={act_val}, Command={command.strip()}")
        # write_log(f"Step {self.lifetime + 1}: Chosen Action={action}, Command={command.strip()}")
        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            write_log(f"❌ Ending episode due to send failure in step: {e}")
            terminated = True
            observation = self.last_observation.copy()
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
                         "reward/step_game_over_penalty": -self.penalty_game_over_coeff, # Log the penalty
                         "env/lines_cleared_this_step": 0,
                         "env/height_increase": 0,
                         "env/hole_increase": 0,
                         "env/current_height": self.current_height,
                         "env/current_holes": self.current_holes,
                         "env/current_lifetime": self.lifetime
                    }
                    wandb.log(log_data) # Log immediately
                except Exception as log_e:
                     if not self._wandb_log_error_reported:
                         print(f"Wandb logging error in step (send fail): {log_e}")
                         self._wandb_log_error_reported = True
            # --- End logging ---

            return observation, reward, terminated, False, info # Return immediately

        # --- Get State Update ---
        terminated, new_lines_removed, new_height, new_holes, observation = self.get_tetris_server_response()

        # --- Calculate Reward ---
        reward = 0.0
        lines_cleared_this_step = new_lines_removed - self.lines_removed

        # --- !!! NEW: Multi-line clear reward logic !!! ---
        line_clear_reward = 0.0
        if lines_cleared_this_step == 1:
            line_clear_reward = 1 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 2:
            line_clear_reward = 4 * self.reward_line_clear_coeff # Quadratic
        elif lines_cleared_this_step == 3:
            line_clear_reward = 9 * self.reward_line_clear_coeff
        elif lines_cleared_this_step >= 4:
            line_clear_reward = 25 * self.reward_line_clear_coeff # Big bonus for Tetris
        reward += line_clear_reward
        # --- END NEW ---

        height_increase = new_height - self.current_height
        height_penalty = 0.0
        if height_increase > 0:
            height_penalty = height_increase * self.penalty_height_increase_coeff
            reward -= height_penalty

        hole_increase = new_holes - self.current_holes
        hole_penalty = 0.0
        if hole_increase > 0:
            hole_penalty = hole_increase * self.penalty_hole_increase_coeff
            reward -= hole_penalty

        step_penalty = self.penalty_step_coeff # Will be 0 if set above
        reward -= step_penalty # Apply step penalty (even if 0)

        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            # Log only once per game over for clarity, ADDED reward breakdown
            write_log(f"💔 Game Over! Final Lines: {new_lines_removed}, Lifetime: {self.lifetime + 1}. Step Reward Breakdown: LC={line_clear_reward:.2f}, HP={-height_penalty:.2f}, OP={-hole_penalty:.2f}, SP={-step_penalty:.2f}, GO={-game_over_penalty:.2f} -> Total={reward:.2f}")
            write_log(f"🔥 Total Episode Reward: {self.episode_total_reward:.2f}")

        self.episode_total_reward += reward
        # --- Update Internal State ---
        self.lines_removed = new_lines_removed
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime}
        truncated = False # DQN typically doesn't use truncation like PPO

        if terminated:
            info['terminal_observation'] = observation.copy()
            # Log final stats here if needed, or use SB3 logger/callback
            # Example: print(f"Episode End: Lines={self.lines_removed}, Lifetime={self.lifetime}, Reward={reward}")


        # --- !!! NEW: Detailed Wandb Logging !!! ---
        if wandb_enabled and run:
             try:
                 log_data = {
                     "reward/step_total": reward,
                     "reward/step_line_clear": line_clear_reward,
                     "reward/step_height_penalty": -height_penalty, # Log penalties as negative values
                     "reward/step_hole_penalty": -hole_penalty,
                     "reward/step_survival_penalty": -step_penalty,
                     "reward/step_game_over_penalty": -game_over_penalty, # Will be non-zero only on last step
                     "env/lines_cleared_this_step": lines_cleared_this_step,
                     "env/height_increase": height_increase,
                     "env/hole_increase": hole_increase,
                     "env/current_height": self.current_height,
                     "env/current_holes": self.current_holes,
                     "env/current_lifetime": self.lifetime # Log lifetime at each step
                 }
                 # Filter out zero reward components (except game over) for cleaner graphs
                 # Keep all env/ metrics
                 filtered_log_data = {k: v for k, v in log_data.items() if not (k.startswith("reward/") and not k.endswith("game_over_penalty") and v == 0) or k.startswith("env/")}
                 # We don't have easy access to the global step here, rely on Wandb/SB3 sync
                 wandb.log(filtered_log_data)
             except Exception as log_e:
                 # Prevent spamming logs if Wandb logging fails repeatedly
                 if not self._wandb_log_error_reported:
                     print(f"Wandb logging error in step: {log_e}")
                     self._wandb_log_error_reported = True
        # --- END NEW ---


        # Optional: Render on step if requested
        if self.render_mode == "human":
              self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the Wandb error reported flag for the new episode
        self._wandb_log_error_reported = False
        self.episode_total_reward = 0.0 

        for attempt in range(3): # Allow a few attempts to reset/reconnect
            try:
                self._send_command(b"start\n")
                terminated, lines, height, holes, observation = self.get_tetris_server_response()
                if terminated:
                    write_log(f"⚠️ Server reported game over on reset attempt {attempt+1}. Retrying...")
                    if attempt < 2: # Reconnect if not last attempt
                        self._connect_socket()
                        time.sleep(0.5) # Small delay before retry
                        continue # Retry the loop
                    else:
                        write_log("❌ Server still terminated after multiple reset attempts. Cannot proceed.")
                        raise RuntimeError("Tetris server failed to reset properly.")
                # Reset successful
                self.lines_removed = 0
                self.current_height = height
                self.current_holes = holes
                self.lifetime = 0
                self.last_observation = observation.copy()
                # write_log(f"🔄 Environment Reset. Initial state: H={height}, O={holes}") # Less verbose logging
                info = {}
                return observation, info

            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError) as e:
                 write_log(f"🔌 Connection issue during reset attempt {attempt+1} ({e}). Retrying...")
                 if attempt < 2:
                     try:
                         self._connect_socket() # Attempt reconnect
                         time.sleep(0.5)
                     except ConnectionError:
                         write_log("   Reconnect failed.")
                         if attempt == 1: # If second attempt also fails, raise
                             raise RuntimeError(f"Failed to reconnect and reset Tetris server after multiple attempts: {e}")
                 else: # Final attempt failed
                     raise RuntimeError(f"Failed to reset Tetris server after multiple attempts: {e}")

        # Should not be reached if logic is correct, but as fallback:
        raise RuntimeError("Failed to reset Tetris server.")


    def render(self):
        self._initialize_pygame() # Ensure pygame is ready if in human mode

        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None:
                 # This should not happen if _initialize_pygame worked, but handle defensively
                 write_log("⚠️ Render called but Pygame window is not initialized.")
                 return

            if hasattr(self, 'last_raw_render_frame'):
                try:
                    # last_raw_render_frame is (H, W, C) BGR from OpenCV
                    render_frame_rgb = cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
                    # Pygame surface requires (width, height)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    # Transpose needed: (H, W, C) -> (W, H, C) for Pygame surfarray
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))
                    # Scale up to window size
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0))
                    pygame.event.pump() # Process internal Pygame events
                    pygame.display.flip() # Update the full screen surface
                    self.clock.tick(self.metadata["render_fps"]) # Control frame rate
                except Exception as e:
                    write_log(f"⚠️ Error during Pygame rendering: {e}")
                    # Attempt to close pygame gracefully on error?
                    # self.close()

            else:
                # Draw a black screen if no frame available yet
                 self.window_surface.fill((0, 0, 0))
                 pygame.display.flip()

        elif self.render_mode == "rgb_array":
             if hasattr(self, 'last_raw_render_frame'):
                 # Return RGB (H, W, C)
                 return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
             else:
                 # Return black frame if no observation yet
                 return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self):
        # write_log("🔌 Closing environment connection.") # Less verbose
        if self.client_sock:
            try:
                self.client_sock.close()
            except socket.error as e:
                 write_log(f"   Error closing socket: {e}")
            self.client_sock = None

        if self.is_pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.is_pygame_initialized = False
                # write_log("   Pygame window closed.") # Less verbose
            except Exception as e:
                 write_log(f"   Error closing Pygame: {e}")

# --- Environment Setup ---
write_log("✅ 建立基礎環境函數 make_env...")
def make_env():
    """Helper function to create an instance of the Tetris environment."""
    env = TetrisEnv()
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

write_log("   環境建立完成並已包裝 (DummyVecEnv -> VecFrameStack -> VecNormalize)")


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
tau = 1.0 # Default for DQN
target_update_interval = current_config.get("target_update_interval", 10000)
gradient_steps = 1 # Default for DQN
# --- !!! UPDATED Exploration Fraction used here !!! ---
exploration_fraction = current_config.get("exploration_fraction", 0.5) # INCREASED default if not in wandb
exploration_final_eps = current_config.get("exploration_final_eps", 0.05)

# Define DQN model
model = DQN(
    policy=policy_type,
    env=train_env,
    verbose=1,
    gamma=gamma_param, # Use loaded gamma
    learning_rate=learning_rate,
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs={
        "buffer_size": 100_000,
        "alpha": 0.6,
        "epsilon": 1e-6,
        # for importance sampling
        "beta": 0.4,
        # other ReplayBuffer args if needed…
    },
    buffer_size=buffer_size,
    learning_starts=learning_starts,
    batch_size=batch_size,
    tau=tau,
    train_freq=(1, "step"), # Train every step
    gradient_steps=gradient_steps,
    target_update_interval=target_update_interval,
    exploration_fraction=exploration_fraction, # Use the updated value
    exploration_final_eps=exploration_final_eps,
    policy_kwargs=dict(normalize_images=True), # As per original code
    seed=42, # Set seed for reproducibility
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log=f"/kaggle/working/runs/{run_id}" if wandb_enabled else None # Log TB only if wandb enabled
)
write_log(f"   模型建立完成. Device: {model.device}")
write_log(f"   使用的超參數: {model.get_parameters()['policy']}") # Log actual params used


# Setup Wandb callback if enabled
if wandb_enabled:
    wandb_callback = WandbCallback(
        gradient_save_freq=10000, # Log grads every 10k steps
        model_save_path=f"/kaggle/working/models/{run_id}", # Save models periodically
        model_save_freq=50000, # Save every 50k steps
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
     write_log(f"❌ 訓練過程中發生錯誤: {e}") # Log exception info
     # Save model before exiting if error occurs mid-training
     error_save_path = f'/kaggle/working/{STUDENT_ID}_dqn_error_save_{run_id}.zip' # Include run_id
     try:
         model.save(error_save_path)
         write_log(f"   模型已嘗試儲存至 {error_save_path}")
         if wandb_enabled and run: wandb.save(error_save_path) # Upload error model to wandb
     except Exception as save_e:
          write_log(f"   ❌ 儲存錯誤模型時也發生錯誤: {save_e}")
     if run: run.finish(exit_code=1, quiet=True) # Finish wandb run with error code

# --- Save Final Model (only if training completed successfully) ---
if training_successful:
    stats_path = f"/kaggle/working/vecnormalize_stats_{run_id}.pkl"
    final_model_name = f'{STUDENT_ID}_dqn_final_{run_id}.zip'
    final_model_path = os.path.join("/kaggle/working", final_model_name)

    try:
        train_env.save(stats_path)
        write_log(f"   VecNormalize 統計數據已儲存至 {stats_path}")
        if wandb_enabled and run: wandb.save(stats_path) # Upload stats to wandb

        model.save(final_model_path)
        write_log(f"✅ 最終模型已儲存: {final_model_path}")
        display(FileLink(final_model_path))
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
    try:
        eval_env_base = DummyVecEnv([make_env])

        # Wrap with FrameStack FIRST, same as training
        # Use wandb config if available, otherwise use default from global config
        n_stack_eval = run.config.get("n_stack", config["n_stack"]) if run else config["n_stack"]
        eval_env_stacked = VecFrameStack(eval_env_base, n_stack=n_stack_eval, channels_order="first")

        # Load the SAME VecNormalize statistics
        eval_env = VecNormalize.load(stats_path, eval_env_stacked)
        eval_env.training = False  # Set mode to evaluation
        eval_env.norm_reward = False # IMPORTANT: দেখতে আসল reward (View actual rewards)

        write_log("   評估環境建立成功.")

    except FileNotFoundError:
        write_log(f"❌ 錯誤: VecNormalize 統計文件未找到於 {stats_path}。跳過評估。")
        eval_env = None
    except Exception as e:
        write_log(f"❌ 建立評估環境時出錯: {e}")
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
                obs = eval_env.reset()
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
                            # Note: get_attr might return a list if using SubprocVecEnv, but here DummyVecEnv has one env
                            base_env = eval_env.get_attr("envs")[0].env
                            raw_frame = base_env.render(mode="rgb_array")
                            if raw_frame is not None:
                                frames.append(raw_frame)
                        except Exception as render_err:
                            write_log(f"⚠️ 評估時獲取渲染幀出錯: {render_err}")

                    # Predict and step using the trained model
                    action, _ = model.predict(obs, deterministic=True) # Use deterministic actions for evaluation
                    obs, reward, done, infos = eval_env.step(action)

                    # Accumulate rewards and stats (remember reward is not normalized here)
                    episode_reward += reward[0] # VecEnv returns lists
                    last_info = infos[0]
                    # Use .get() for safety, default to previous value if key missing
                    # Ensure correct keys from TetrisEnv's info dict
                    episode_lines = last_info.get('removed_lines', episode_lines)
                    episode_lifetime = last_info.get('lifetime', episode_lifetime)
                    done = done[0] # VecEnv returns lists

                write_log(f"   評估 Episode {i+1}: Reward={episode_reward:.2f}, Lines={episode_lines}, Steps={episode_lifetime}")
                total_rewards.append(episode_reward)
                total_lines.append(episode_lines)
                total_lifetimes.append(episode_lifetime)
                if i == 0: all_frames = frames # Store frames from the first episode

            write_log(f"--- 評估結果 ({num_eval_episodes} episodes) ---")
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            mean_lines = np.mean(total_lines)
            std_lines = np.std(total_lines)
            mean_lifetime = np.mean(total_lifetimes)
            std_lifetime = np.std(total_lifetimes)

            write_log(f"   平均 Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            write_log(f"   平均 Lines: {mean_lines:.2f} +/- {std_lines:.2f}")
            write_log(f"   平均 Steps: {mean_lifetime:.2f} +/- {std_lifetime:.2f}")

            # Log evaluation metrics to Wandb
            if wandb_enabled and run:
                wandb.log({
                    "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                    "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                    "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime,
                })

            # --- Generate Replay GIF ---
            if all_frames:
                gif_path = f'/kaggle/working/replay_eval_{run_id}.gif'
                write_log(f"💾 正在儲存評估回放 GIF 至 {gif_path}...")
                try:
                    imageio.mimsave(gif_path, [np.array(frame).astype(np.uint8) for frame in all_frames if frame is not None], fps=15, loop=0)
                    write_log("   GIF 儲存成功.")
                    display(FileLink(gif_path))
                    if wandb_enabled and run: wandb.log({"eval/replay": wandb.Video(gif_path, fps=15, format="gif")}) # Log GIF to Wandb
                except Exception as e: write_log(f"   ❌ 儲存 GIF 時發生錯誤: {e}")
            else: write_log("   ⚠️ 未能儲存 GIF (沒有收集到幀或第一輪評估出錯).")

            # --- Save Evaluation Results CSV ---
            csv_filename = f'tetris_evaluation_scores_{run_id}.csv'
            csv_path = os.path.join("/kaggle/working", csv_filename)
            try:
                with open(csv_path, 'w') as fs:
                    fs.write('episode_id,removed_lines,played_steps,reward\n')
                    if total_lines: # Ensure lists are not empty before accessing index 0
                         for i in range(len(total_lines)):
                             fs.write(f'eval_{i},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                    fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                write_log(f"✅ 評估分數 CSV 已儲存: {csv_path}")
                display(FileLink(csv_path))
                if wandb_enabled and run: wandb.save(csv_path) # Upload CSV to wandb
            except Exception as e: write_log(f"   ❌ 儲存 CSV 時發生錯誤: {e}")

        except Exception as eval_e:
            write_log(f"❌ 評估迴圈中發生錯誤: {eval_e}", exc_info=True)

        finally:
             # Ensure evaluation env is closed even if errors occur
             if eval_env:
                 eval_env.close()
                 write_log("   評估環境已關閉.")

# --- Cleanup ---
write_log("🧹 清理環境...")
if 'train_env' in locals() and train_env: # Check if train_env exists and is not None
    try:
        train_env.close()
        write_log("   訓練環境已關閉.")
    except Exception as e:
        write_log(f"   關閉訓練環境時出錯: {e}")

# Close the Java server process
if java_process and java_process.poll() is None: # Check if process exists and is running
     write_log("   正在終止 Java server process...")
     java_process.terminate()
     try:
         java_process.wait(timeout=5) # Wait up to 5 seconds
         write_log("   Java server process 已終止.")
     except subprocess.TimeoutExpired:
         write_log("   Java server 未能在 5 秒內終止, 強制結束...")
         java_process.kill()
         write_log("   Java server process 已強制結束.")
elif java_process and java_process.poll() is not None:
     write_log("   Java server process 已自行結束.")
else:
     write_log("   Java server process 未啟動或已關閉.")


# Finish the Wandb run if it was initialized and training didn't crash early
if run: # Check if run object exists
    if training_successful:
         run.finish()
         write_log("✨ Wandb run finished.")
    else:
         # Run might have already been finished in the exception handler
         # Check run.is_running before finishing again
         if hasattr(run, 'is_running') and run.is_running:
             run.finish(exit_code=1) # Ensure it's marked as failed
             write_log("✨ Wandb run finished (marked as failed due to error).")
         elif not hasattr(run, 'is_running'): # Fallback for older wandb versions?
             try: # Try finishing anyway, might raise if already finished
                 run.finish(exit_code=1)
                 write_log("✨ Wandb run finished (marked as failed due to error - fallback).")
             except Exception:
                 write_log("✨ Wandb run likely already finished (marked as failed due to error).")


write_log("🏁 腳本執行完畢.")