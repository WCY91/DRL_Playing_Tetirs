# -*- coding: utf-8 -*-
import numpy as np
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
from stable_baselines3.common.callbacks import BaseCallback # Import BaseCallback
from IPython.display import FileLink, display
import torch
import time
import pygame
import traceback # Import traceback for detailed error logging

# --- Wandb Setup ---
# Handle Kaggle secrets loading more robustly
try:
    from kaggle_secrets import UserSecretsClient
    secrets_available = True
except ImportError:
    secrets_available = False
    print("Kaggle secrets not available. Set WANDB_API_KEY environment variable manually if needed.")

from wandb.integration.sb3 import WandbCallback

# --- Configuration ---
STUDENT_ID = "113598065"
TOTAL_TIMESTEPS = 1000000 # Adjust as needed

# --- Wandb Login ---
# ================================================================
# !! 重要提示 (Wandb Login Issue) !!
# 您的 Log 顯示 "Wandb login failed: Unexpected response... No user secrets exist for kernel id..."
# 這表示程式無法從 Kaggle Secrets 或環境變數中正確讀取您的 WANDB_API_KEY。
# 這通常與 Kaggle 環境或您的 Secrets 設定有關，而不是程式碼邏輯錯誤。
#
# 請確認：
# 1. 您是否在 Kaggle Notebook 的 "Add-ons" -> "Secrets" 中正確設定了名為 "WANDB_API_KEY" 的 Secret？
# 2. 如果不是在 Kaggle，您是否已將 WANDB_API_KEY 設置為環境變數？
#
# 如果無法解決，Wandb 日誌記錄將無法使用 (如 Log 中所示 "Running without Wandb logging.")。
# 您仍然可以訓練模型，但無法在 Wandb 網站上追蹤實驗過程。
# ================================================================
wandb_enabled = False
WANDB_API_KEY = None
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
        # Add timeout to login
        try:
            wandb.login(key=WANDB_API_KEY, timeout=30) # Increased timeout
            wandb_enabled = True
            print("✅ Wandb login successful.")
        except Exception as login_e:
             print(f"Wandb login attempt failed: {login_e}. Running without Wandb logging.")
             wandb_enabled = False
             WANDB_API_KEY = None # Ensure it's None on failure
    else:
        print("WANDB_API_KEY not found in Kaggle secrets or environment variables. Running without Wandb logging.")

except Exception as e:
    # Catch potential UserSecretsClient errors etc.
    print(f"Wandb setup failed during secret retrieval: {e}. Running without Wandb logging.")
    wandb_enabled = False
    WANDB_API_KEY = None # Ensure it's None on failure


# --- Config - MODIFIED for Curriculum Learning ---
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1-Curriculum", # New ID reflecting the change
    # --- PPO Specific Params (Stabilized) ---
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.15,         # Keep some exploration pressure
    "learning_rate": 1e-4,    # Learning rate
    # --- Common Params ---
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Reward Coeffs (Base values) ---
    "reward_line_clear_coeff": 500.0,    # Keep high line clear reward
    "penalty_height_increase_coeff": 0.5, # Base height penalty
    "penalty_hole_increase_coeff": 0.5,   # Base hole penalty
    "penalty_step_coeff": 0.1,            # Small survival reward per step
    # --- Curriculum Learning Params ---
    "penalty_game_over_start_coeff": 1.0,   # <<< MODIFIED: Start with very low GO penalty
    "penalty_game_over_end_coeff": 50.0,    # <<< MODIFIED: Target GO penalty
    "curriculum_anneal_fraction": 0.5,      # <<< NEW: Increase GO penalty over the first 50% of training
}

# --- Wandb Init ---
run = None # Initialize run to None
project_name = "tetris-training-curriculum" # Update project name
run_id = f"local_ppo_curriculum_{int(time.time())}" # Default run_id
if wandb_enabled:
    try:
        run = wandb.init(
            project=project_name,
            entity="t113598065-ntut-edu-tw", # Replace with your Wandb entity if needed
            sync_tensorboard=True, monitor_gym=True, save_code=True,
            settings=Settings(init_timeout=180), config=config,
            id=run_id, # Use generated run_id
            resume="allow" # Allow resuming if run_id exists
        )
        run_id = run.id # Update run_id if init is successful
        print(f"✅ Wandb run initialized successfully. Project: {project_name}, Run ID: {run_id}")
    except Exception as e:
        print(f"Wandb initialization failed: {e}. Disabling Wandb.")
        wandb_enabled = False
        run = None # Ensure run is None if init fails

log_path = f"/kaggle/working/tetris_train_log_{run_id}.txt"

# --- Helper Functions & Server Start ---
def write_log(message, exc_info=False):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    print(log_message)
    if exc_info:
        log_message += "\n" + traceback.format_exc() # Use imported traceback
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f: f.write(log_message + "\n")
    except Exception as e: print(f"Error writing log to {log_path}: {e}")

def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60):
    write_log(f"⏳ Waiting for Tetris TCP server @ {ip}:{port}...")
    start_time = time.time()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0); s.connect((ip, port))
            write_log("✅ Java TCP server ready."); return True
        except socket.error:
            if time.time() - start_time > timeout: write_log(f"❌ Timeout waiting for server ({timeout}s)"); return False
            time.sleep(1.0)

java_process = None
try:
    write_log("🚀 Attempting to start Java Tetris server...")
    jar_file = "TetrisTCPserver_v0.6.jar"
    if not os.path.exists(jar_file): raise FileNotFoundError(f"JAR file not found: '{jar_file}' at '{os.getcwd()}'")
    java_process = subprocess.Popen(["java", "-jar", jar_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_log(f"✅ Java server process started (PID: {java_process.pid})")
    if not wait_for_tetris_server(): raise TimeoutError("Java server did not become available.")
except Exception as e:
    write_log(f"❌ Error starting/waiting for Java server: {e}", exc_info=True)
    if java_process and java_process.poll() is None:
        write_log("   Terminating Java process..."); java_process.terminate()
        try: java_process.wait(timeout=2)
        except subprocess.TimeoutExpired: java_process.kill()
    raise # Re-raise the exception to stop the script if server fails

if torch.cuda.is_available(): write_log(f"✅ PyTorch using GPU: {torch.cuda.get_device_name(0)}")
else: write_log("⚠️ PyTorch using CPU.")

# ----------------------------
# 定義 Tetris 環境 (整合 Curriculum Learning)
# ----------------------------
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris (Curriculum Learning Focus)."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS = 5
    IMG_HEIGHT = 200; IMG_WIDTH = 100; IMG_CHANNELS = 3
    RESIZED_DIM = 84

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None, env_config=None): # Pass config
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), dtype=np.uint8)
        self.server_ip = host_ip; self.server_port = host_port
        self.client_sock = None; self._connect_socket()

        self.current_cumulative_lines = 0
        self.current_height = 0; self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None

        # Use passed config or global config as fallback
        current_config = env_config if env_config is not None else config
        self.reward_line_clear_coeff = current_config.get("reward_line_clear_coeff", 500.0)
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 0.5)
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 0.5)
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.1)

        # <<< MODIFIED: Initialize with the STARTING value for game over penalty >>>
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_start_coeff", 1.0) # Use start coeff
        self.current_go_penalty = self.penalty_game_over_coeff # Store current value separately for logging

        write_log(f"TetrisEnv initialized (Curriculum Learning Focus).")
        write_log(f"Initial Reward Coeffs: LC={self.reward_line_clear_coeff}, H_pen={self.penalty_height_increase_coeff}, O_pen={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO_pen_Initial={self.penalty_game_over_coeff}")

        self.window_surface = None; self.clock = None
        self.is_pygame_initialized = False; self._wandb_log_error_reported = False

    # <<< NEW: Method to update game over penalty dynamically >>>
    def set_game_over_penalty(self, new_penalty_value):
        """Allows the game over penalty coefficient to be updated externally (e.g., by a callback)."""
        # Optional: Add some logging if the value changes significantly
        # if abs(new_penalty_value - self.penalty_game_over_coeff) > 0.1:
        #     write_log(f"  Updating GO Penalty from {self.penalty_game_over_coeff:.2f} to {new_penalty_value:.2f}")
        self.penalty_game_over_coeff = new_penalty_value
        self.current_go_penalty = new_penalty_value # Update for logging

    def _initialize_pygame(self):
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                import pygame; pygame.init(); pygame.display.init()
                self.window_surface = pygame.display.set_mode((self.RESIZED_DIM * 4, self.RESIZED_DIM * 4))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock(); self.is_pygame_initialized = True; write_log("   Pygame initialized.")
            except Exception as e: write_log(f"⚠️ Error initializing Pygame: {e}"); self.render_mode = None

    def _connect_socket(self):
        try:
            if self.client_sock:
                try: self.client_sock.close()
                except socket.error: pass
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0); self.client_sock.connect((self.server_ip, self.server_port))
        except socket.error as e: raise ConnectionError(f"Failed connect @ {self.server_ip}:{self.server_port}: {e}")

    def _send_command(self, command: bytes):
        if not self.client_sock: raise ConnectionError("Socket disconnected.")
        try: self.client_sock.sendall(command)
        except socket.timeout: raise ConnectionAbortedError("Socket timeout on send")
        except socket.error as e: raise ConnectionAbortedError(f"Socket error on send: {e}")

    def _receive_data(self, size):
        if not self.client_sock: raise ConnectionError("Socket disconnected.")
        data = b""; self.client_sock.settimeout(10.0)
        try:
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk: raise ConnectionAbortedError("Socket broken (received empty chunk)")
                data += chunk
        except socket.timeout: raise ConnectionAbortedError(f"Socket timeout receiving {size} bytes (got {len(data)})")
        except socket.error as e: raise ConnectionAbortedError(f"Socket error on receive: {e}")
        return data

    def get_tetris_server_response(self):
        try:
            is_game_over = (self._receive_data(1) == b'\x01')
            cumulative_lines = int.from_bytes(self._receive_data(4), 'big')
            height = int.from_bytes(self._receive_data(4), 'big')
            holes = int.from_bytes(self._receive_data(4), 'big')
            img_size = int.from_bytes(self._receive_data(4), 'big')
            if img_size <= 0 or img_size > 1000000: raise ValueError(f"Invalid img size: {img_size}")
            img_png = self._receive_data(img_size)
            nparr = np.frombuffer(img_png, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if np_image is None:
                write_log("❌ Image decode failed. Using last valid observation."); return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8)
            self.last_raw_render_frame = resized.copy()
            self.last_observation = observation.copy()
            return is_game_over, cumulative_lines, height, holes, observation
        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e:
            write_log(f"❌ Connection/Value error getting response: {e}. Ending episode.")
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"❌ Unexpected error getting response: {e}. Ending episode.", exc_info=True)
            return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action):
        command_map = {0: b"move -1\n", 1: b"move 1\n", 2: b"rotate 0\n", 3: b"rotate 1\n", 4: b"drop\n"}
        command = command_map.get(action, b"drop\n")
        if action not in command_map: write_log(f"⚠️ Invalid action: {action}.")

        try:
            self._send_command(command)
            terminated, next_cumulative_lines, new_height, new_holes, observation = self.get_tetris_server_response()
        except (ConnectionAbortedError, ConnectionError, ValueError) as e:
            write_log(f"❌ Error during send/receive in step: {e}. Ending episode.")
            # Apply the CURRENT game over penalty even on comm error
            reward = -self.penalty_game_over_coeff # Use the potentially low early penalty
            info = {'removed_lines': self.current_cumulative_lines, 'lifetime': self.lifetime, 'final_status': 'comm_error'}
            info['terminal_observation'] = self.last_observation.copy()
            if wandb_enabled and run: self._safe_wandb_log({"reward/step_total": reward, "reward/step_game_over_penalty": -self.penalty_game_over_coeff})
            return self.last_observation.copy(), reward, True, True, info # Terminated=True, Truncated=True

        lines_cleared_this_step = next_cumulative_lines - self.current_cumulative_lines
        reward = 0.0; line_clear_reward = 0.0

        # Line Clear Reward (Quadratic bonus)
        if lines_cleared_this_step == 1: line_clear_reward = 1 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 2: line_clear_reward = 4 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 3: line_clear_reward = 9 * self.reward_line_clear_coeff
        elif lines_cleared_this_step >= 4: line_clear_reward = 25 * self.reward_line_clear_coeff
        reward += line_clear_reward

        # Height Penalty
        height_increase = max(0, new_height - self.current_height)
        height_penalty = height_increase * self.penalty_height_increase_coeff
        reward -= height_penalty

        # Hole Penalty
        hole_increase = max(0, new_holes - self.current_holes)
        hole_penalty = hole_increase * self.penalty_hole_increase_coeff
        reward -= hole_penalty

        # Survival Reward
        step_reward_value = self.penalty_step_coeff
        reward += step_reward_value

        game_over_penalty = 0.0
        if terminated:
            # <<< MODIFIED: Use the CURRENT, dynamically adjusted game over penalty >>>
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            # Log using the current penalty value for clarity
            write_log(f"💔 GameOver L={next_cumulative_lines} T={self.lifetime+1} | Rews: LC={line_clear_reward:.1f} HP={-height_penalty:.1f} OP={-hole_penalty:.1f} Step={step_reward_value:.1f} GO={-game_over_penalty:.1f} (Current) -> Tot={reward:.1f}")

        self.current_cumulative_lines = next_cumulative_lines
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        truncated = False
        info = {'removed_lines': self.current_cumulative_lines, 'lifetime': self.lifetime}
        if terminated:
            info['terminal_observation'] = observation.copy()
            info['episode'] = { # For SB3 logging
                'r': reward,
                'l': self.lifetime,
                'lines': self.current_cumulative_lines,
                'current_go_penalty': game_over_penalty # Log the penalty applied at game end
             }

        # Log details to Wandb (if enabled)
        if wandb_enabled and run:
            log_data = {
                "reward/step_total": reward,
                "reward/step_line_clear": line_clear_reward,
                "reward/step_height_penalty": -height_penalty,
                "reward/step_hole_penalty": -hole_penalty,
                "reward/step_survival_reward": step_reward_value,
                "reward/step_game_over_penalty": -game_over_penalty if terminated else 0.0,
                "env/lines_cleared_this_step": lines_cleared_this_step,
                "env/height_increase": height_increase,
                "env/hole_increase": hole_increase,
                "env/current_height": self.current_height,
                "env/current_holes": self.current_holes,
                "env/current_lifetime": self.lifetime,
                "env/current_go_penalty_coeff": self.current_go_penalty # Log current penalty coeff every step
            }
            self._safe_wandb_log(log_data)

        if self.render_mode == "human": self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._wandb_log_error_reported = False
        for attempt in range(3):
            try:
                self._connect_socket()
                self._send_command(b"start\n")
                terminated, server_lines, height, holes, observation = self.get_tetris_server_response()
                if terminated or server_lines != 0:
                    write_log(f"⚠️ Invalid server state on reset attempt {attempt+1} (Terminated={terminated}, Lines={server_lines}). Retrying...")
                    if attempt < 2: time.sleep(0.5 + attempt * 0.5); continue
                    else: raise RuntimeError("Server failed to provide a valid reset state after multiple attempts.")

                self.current_cumulative_lines = 0; self.current_height = height
                self.current_holes = holes; self.lifetime = 0
                self.last_observation = observation.copy()
                info = {'start_height': height, 'start_holes': holes}
                return observation, info
            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError, ValueError) as e:
                 write_log(f"🔌 Connection/Value issue during reset attempt {attempt+1}. Retrying... ({e})")
                 if attempt < 2:
                     time.sleep(1.0 + attempt * 0.5)
                     if self.client_sock:
                         try: self.client_sock.close()
                         except: pass
                         self.client_sock = None
                 else: raise RuntimeError(f"Failed to reset environment after multiple connection attempts: {e}")
            except Exception as e:
                 write_log(f"❌ Unexpected error during reset attempt {attempt+1}: {e}", exc_info=True)
                 if attempt < 2: time.sleep(1.0 + attempt * 0.5)
                 else: raise RuntimeError(f"Failed reset due to unexpected error: {e}")
        raise RuntimeError("Failed to reset environment after retry loop.")

    def render(self):
        self._initialize_pygame()
        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None: return
            current_frame = self.last_raw_render_frame
            if current_frame is not None:
                try:
                    render_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0)); pygame.event.pump()
                    pygame.display.flip(); self.clock.tick(self.metadata["render_fps"])
                except Exception as e: write_log(f"⚠️ Pygame render error: {e}")
            else:
                try: self.window_surface.fill((0,0,0)); pygame.display.flip()
                except Exception as e: write_log(f"⚠️ Pygame fill error: {e}")
        elif self.render_mode == "rgb_array":
             if self.last_raw_render_frame is not None: return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
             else: return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self):
        if self.client_sock:
            try: self.client_sock.close(); write_log("   Socket closed.")
            except socket.error: pass
            self.client_sock = None
        if self.is_pygame_initialized:
            try: import pygame; pygame.display.quit(); pygame.quit(); write_log("   Pygame closed.")
            except Exception: pass
            self.is_pygame_initialized = False

    def _safe_wandb_log(self, data):
        if wandb_enabled and run:
            try:
                wandb.log(data)
            except Exception as log_e:
                if not self._wandb_log_error_reported:
                    write_log(f"⚠️ Wandb log error: {log_e}")
                    self._wandb_log_error_reported = True

# ----------------------------
# Curriculum Learning Callback
# ----------------------------
class CurriculumCallback(BaseCallback):
    """
    A custom callback to gradually increase the game over penalty during training.
    """
    def __init__(self,
                 penalty_start: float,
                 penalty_end: float,
                 anneal_fraction: float,
                 total_training_steps: int,
                 verbose: int = 0):
        super(CurriculumCallback, self).__init__(verbose)
        self.penalty_start = penalty_start
        self.penalty_end = penalty_end
        self.anneal_fraction = anneal_fraction
        self.total_training_steps = total_training_steps
        self.anneal_timesteps = int(total_training_steps * anneal_fraction)
        write_log(f"CurriculumCallback initialized: GO Penalty from {penalty_start:.2f} to {penalty_end:.2f} over {self.anneal_timesteps} steps ({anneal_fraction*100:.0f}% of training).")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Calculate current progress fraction through the annealing phase
        progress = min(1.0, self.num_timesteps / self.anneal_timesteps)

        # Linearly interpolate the penalty
        current_penalty = self.penalty_start + progress * (self.penalty_end - self.penalty_start)

        # Update the penalty in each environment instance within the VecEnv
        # Important: Assumes DummyVecEnv or SubprocVecEnv where env_method is available
        try:
            # env_method calls the 'set_game_over_penalty' method on each underlying env
            self.training_env.env_method('set_game_over_penalty', current_penalty)

            # Log the current penalty value to Wandb (and console via verbose>0 in callback)
            if self.num_timesteps % 1000 == 0: # Log every 1000 steps to avoid too much noise
                 self.logger.record('train/current_go_penalty_coeff', current_penalty)
                 if self.verbose > 1 :
                     print(f" Timestep {self.num_timesteps}: GO Penalty updated to {current_penalty:.2f}")

        except AttributeError:
            # Handle cases where env_method might not be available (e.g., non-vectorized env)
            if self.verbose > 0:
                print("Warning: Could not call 'env_method' on training_env. Curriculum penalty update skipped.")
        except Exception as e:
             if self.verbose > 0:
                print(f"Warning: Error calling env_method in CurriculumCallback: {e}")

        return True # Continue training

# --- Environment Setup ---
write_log("✅ Creating env function (Curriculum Learning Focus)...")
def make_env(env_config=None): # Pass config to env constructor
    # Pass the full config dict so the env can read necessary parameters
    env = TetrisEnv(render_mode=None, env_config=env_config)
    return env

write_log("✅ Creating Vec Env (Dummy)...")
# Use a lambda to ensure a new env instance is created
# Pass the config dictionary to the make_env function within the lambda
train_env_base = DummyVecEnv([lambda: make_env(env_config=config)]) # Pass config here

write_log("✅ Wrapping Env (VecFrameStack)...")
n_stack_param = config.get("n_stack", 4)
train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack_param, channels_order="first")

write_log("✅ Wrapping Env (VecNormalize - Rewards Normalized)...")
gamma_param = config.get("gamma", 0.99)
# Normalize rewards during training as scales differ significantly
train_env = VecNormalize(train_env_stacked,
                         norm_obs=False,       # Observations are already 0-255 uint8
                         norm_reward=True,     # Normalize rewards
                         gamma=gamma_param,
                         clip_reward=10.0)     # Clip rewards
write_log(f"   Environment setup complete (NormReward={train_env.norm_reward}, ClipReward={train_env.clip_reward}).")


# ----------------------------
# PPO Model Setup and Training
# ----------------------------
write_log("🧠 Setting up PPO model...")
clip_range_val = config.get("clip_range", 0.1)
if callable(clip_range_val):
    clip_range_val = clip_range_val(1.0)

ppo_params = {
    "policy": config["policy_type"],
    "env": train_env,
    "verbose": 1,
    "gamma": config["gamma"],
    "learning_rate": config["learning_rate"],
    "n_steps": config["n_steps"],
    "batch_size": config["batch_size"],
    "n_epochs": config["n_epochs"],
    "gae_lambda": config["gae_lambda"],
    "clip_range": clip_range_val,
    "ent_coef": config["ent_coef"],
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tensorboard_log": f"/kaggle/working/runs/{run_id}" if wandb_enabled and run else None,
    "policy_kwargs": dict(normalize_images=False) # Obs already normalized or uint8
}

model = PPO(**ppo_params)

write_log(f"   PPO model created. Device: {model.device}")
write_log(f"   Key Config: lr={config['learning_rate']:.1e}, ent={config['ent_coef']:.2f}, H_pen={config['penalty_height_increase_coeff']:.1f}, O_pen={config['penalty_hole_increase_coeff']:.1f}, StepRew={config['penalty_step_coeff']:.1f}")
write_log(f"   Curriculum: GO_Pen Start={config['penalty_game_over_start_coeff']:.1f}, End={config['penalty_game_over_end_coeff']:.1f}, Anneal={config['curriculum_anneal_fraction']*100:.0f}%")
write_log(f"   Key PPO Params: vf={model.vf_coef}, grad_norm={model.max_grad_norm}")
write_log(f"   Key PPO Clip Range (config): {clip_range_val:.2f}")
write_log(f"   VecNormalize: norm_reward={train_env.norm_reward}, norm_obs={train_env.norm_obs}")


# --- Callbacks ---
callback_list = []

# <<< MODIFIED: Add Curriculum Callback >>>
curriculum_callback = CurriculumCallback(
    penalty_start=config["penalty_game_over_start_coeff"],
    penalty_end=config["penalty_game_over_end_coeff"],
    anneal_fraction=config["curriculum_anneal_fraction"],
    total_training_steps=TOTAL_TIMESTEPS,
    verbose=1 # Set to 1 or 2 for more log messages from callback
)
callback_list.append(curriculum_callback)
write_log("   CurriculumCallback added.")


if wandb_enabled and run:
    try:
        wandb_callback = WandbCallback(
              model_save_path=f"/kaggle/working/models/{run_id}",
              model_save_freq=50000, # Save every 50k steps
              log="all", # Log gradients and histograms
              verbose=2 # Print messages when saving models
           )
        callback_list.append(wandb_callback)
        write_log("   WandbCallback added.")
    except Exception as e:
        write_log(f"⚠️ Error creating WandbCallback: {e}")

# Add other callbacks if needed (e.g., CheckpointCallback, EvalCallback)

# --- Training ---
write_log(f"🚀 Starting PPO training for {TOTAL_TIMESTEPS} steps (Curriculum Learning Focus)...")
training_successful = False
try:
    # log_interval=1 logs Tensorboard scalars every episode
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, log_interval=1)
    write_log("✅ Training complete!")
    training_successful = True
except KeyboardInterrupt:
     write_log("🛑 Training interrupted by user.")
     training_successful = False # Mark as not fully successful
except Exception as e:
    write_log(f"❌ Training error: {e.__class__.__name__}: {e}", exc_info=True)
    training_successful = False
    error_save_path = f'/kaggle/working/{STUDENT_ID}_ppo_curriculum_err_{run_id}.zip'
    try:
        model.save(error_save_path)
        write_log(f"   Attempted to save model on error to {error_save_path}")
        if wandb_enabled and run: wandb.save(error_save_path, base_path="/kaggle/working"); write_log("   Error model uploaded to Wandb.")
    except Exception as save_e: write_log(f"   ❌ Error saving model after training error: {save_e}")
finally:
    # Ensure Wandb run is finished properly
    if wandb_enabled and run:
        exit_code = 1
        if 'training_successful' in locals() and training_successful:
            exit_code = 0
        try:
            # Check if run is still active before finishing
            if wandb.run and wandb.run.id == run.id:
                run.finish(exit_code=exit_code, quiet=True)
                write_log(f"   Wandb run finished with exit code {exit_code}.")
            else:
                write_log("   Wandb run already finished or not active.")
            run = None # Set run to None after finishing
        except Exception as finish_e:
            write_log(f"   ⚠️ Error finishing Wandb run: {finish_e}. It might have already finished.")


# --- Save Final Model and Stats ---
# Use a more descriptive name reflecting the strategy
final_model_base_name = f'{STUDENT_ID}_ppo_curriculum_final_{run_id}'
stats_path = f"/kaggle/working/vecnormalize_stats_{final_model_base_name}.pkl"
final_model_path = f"/kaggle/working/{final_model_base_name}.zip"

if training_successful:
    try:
        # Save VecNormalize statistics
        train_env.save(stats_path)
        write_log(f"✅ VecNormalize stats saved to {stats_path}")
        if wandb_enabled and wandb.run: # Check if wandb is still active
             try: wandb.save(stats_path, base_path="/kaggle/working"); write_log("   VecNormalize stats uploaded.")
             except Exception as upload_e: write_log(f"   ⚠️ Error uploading stats: {upload_e}")

        # Save the final PPO model
        model.save(final_model_path)
        write_log(f"✅ Final model saved: {final_model_path}")
        display(FileLink(final_model_path))
        if wandb_enabled and wandb.run:
             try: wandb.save(final_model_path, base_path="/kaggle/working"); write_log("   Final model uploaded.")
             except Exception as upload_e: write_log(f"   ⚠️ Error uploading final model: {upload_e}")

    except Exception as e:
        write_log(f"❌ Error saving final model/stats: {e}", exc_info=True)
        training_successful = False # Mark as failed if saving fails
else:
     write_log("⏩ Skipping final model/stats saving due to incomplete/failed training.")


# ----------------------------
# Evaluation (Conditional)
# ----------------------------
if training_successful and os.path.exists(final_model_path) and os.path.exists(stats_path):
    write_log("\n🧪 Starting evaluation...")
    eval_env = None
    can_evaluate = False
    try:
        def make_eval_env(env_config=None): # Pass config if needed by env init
            env = TetrisEnv(render_mode="rgb_array", env_config=env_config) # Pass config
            return env

        # Pass the config dictionary when creating the eval env function
        eval_env_base = DummyVecEnv([lambda: make_eval_env(env_config=config)]) # Pass config

        n_stack_eval = config.get("n_stack", 4)
        eval_env_stacked = VecFrameStack(eval_env_base, n_stack=n_stack_eval, channels_order="first")

        # Load stats, set training=False, norm_reward=False
        eval_env = VecNormalize.load(stats_path, eval_env_stacked)
        eval_env.training = False
        eval_env.norm_reward = False
        # <<< IMPORTANT: Set eval env's GO penalty to the FINAL value used in training >>>
        eval_env.env_method('set_game_over_penalty', config["penalty_game_over_end_coeff"])
        write_log(f"   Eval env created, loaded stats, and GO penalty set to final value ({config['penalty_game_over_end_coeff']:.1f}).")
        can_evaluate = True

    except Exception as e:
        write_log(f"❌ Error creating evaluation environment: {e}", exc_info=True)
        can_evaluate = False

    if can_evaluate and eval_env is not None:
        num_eval_episodes = 5
        total_rewards, total_lines, total_lifetimes, all_frames = [], [], [], []

        try:
            for i in range(num_eval_episodes):
                obs = eval_env.reset()
                done = False
                ep_rew, ep_lines, ep_len = 0.0, 0, 0
                frames = [] # Only for first episode GIF
                while not done:
                    if i == 0: # Capture frames for GIF
                        try:
                            raw_frame = eval_env.render()
                            if raw_frame is not None and isinstance(raw_frame, np.ndarray):
                                frames.append(raw_frame)
                        except Exception as render_err:
                            if i == 0 and len(frames) % 50 == 0:
                                write_log(f"   (Eval Ep 0) Render error occurred: {render_err}")
                            pass

                    action, _ = model.predict(obs, deterministic=True)

                    try:
                        obs, reward, terminated, truncated, infos = eval_env.step(action)
                        current_reward = reward[0]
                        ep_rew += current_reward
                        done = terminated[0] or truncated[0]
                        info = infos[0]
                        # Get final stats from info if the episode ended
                        if done and 'episode' in info:
                             ep_lines = info['episode'].get('lines', ep_lines)
                             ep_len = info['episode'].get('l', ep_len)
                        else: # Otherwise, update from running info
                             ep_lines = info.get('removed_lines', ep_lines)
                             ep_len = info.get('lifetime', ep_len)

                    except Exception as step_err:
                        write_log(f"❌ Error during eval step (episode {i+1}): {step_err}. Ending episode.", exc_info=True)
                        done = True # Force end episode

                write_log(f"   Eval Episode {i+1}: Reward={ep_rew:.2f}, Lines={ep_lines}, Steps={ep_len}")
                total_rewards.append(ep_rew)
                total_lines.append(ep_lines)
                total_lifetimes.append(ep_len)
                if i == 0: all_frames = frames

            # --- Calculate & Log Averages ---
            if total_rewards:
                mean_reward=np.mean(total_rewards); std_reward=np.std(total_rewards)
                mean_lines=np.mean(total_lines); std_lines=np.std(total_lines)
                mean_lifetime=np.mean(total_lifetimes); std_lifetime=np.std(total_lifetimes)
                write_log(f"--- Evaluation Results ({num_eval_episodes} episodes) ---")
                write_log(f"   Avg Reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
                write_log(f"   Avg Lines:   {mean_lines:.2f} +/- {std_lines:.2f}")
                write_log(f"   Avg Steps:   {mean_lifetime:.2f} +/- {std_lifetime:.2f}")

                if wandb_enabled and wandb.run:
                     try:
                         wandb.log({
                             "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                             "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                             "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime
                         })
                         write_log("   Evaluation metrics logged to Wandb.")
                     except Exception as log_e: write_log(f"   ⚠️ Error logging eval metrics to Wandb: {log_e}")

                # --- Generate GIF ---
                if all_frames:
                    gif_path = f'/kaggle/working/replay_eval_{final_model_base_name}.gif'
                    write_log(f"💾 Saving evaluation GIF: {gif_path}...")
                    try:
                        valid_frames = [f.astype(np.uint8) for f in all_frames if isinstance(f, np.ndarray) and f.size > 0]
                        if valid_frames:
                            imageio.mimsave(gif_path, valid_frames, fps=10, loop=0)
                            write_log("   GIF saved.")
                            display(FileLink(gif_path))
                            if wandb_enabled and wandb.run:
                                try:
                                    wandb.log({"eval/replay": wandb.Video(gif_path, fps=10, format="gif")})
                                    write_log("   GIF uploaded to Wandb.")
                                except Exception as upload_e: write_log(f"   ⚠️ Error uploading GIF to Wandb: {upload_e}")
                        else: write_log("   ⚠️ No valid frames captured for GIF.")
                    except Exception as e: write_log(f"   ❌ Error saving GIF: {e}", exc_info=True)
                else: write_log("   ⚠️ No frames captured for GIF (only done for first eval episode).")

                # --- Save Evaluation Scores CSV ---
                csv_path = f'/kaggle/working/tetris_eval_scores_{final_model_base_name}.csv'
                try:
                    with open(csv_path, 'w') as fs:
                        fs.write('episode_id,removed_lines,played_steps,reward\n')
                        for i in range(len(total_lines)):
                            fs.write(f'eval_{i+1},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                        fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                    write_log(f"✅ Evaluation scores CSV saved: {csv_path}")
                    display(FileLink(csv_path))
                    if wandb_enabled and wandb.run:
                        try: wandb.save(csv_path, base_path="/kaggle/working"); write_log("   CSV uploaded to Wandb.")
                        except Exception as upload_e: write_log(f"   ⚠️ Error uploading CSV to Wandb: {upload_e}")
                except Exception as e: write_log(f"   ❌ Error saving evaluation CSV: {e}")

            else: write_log("   ⚠️ No evaluation episodes completed, cannot calculate averages.")

        except Exception as eval_e:
            write_log(f"❌ Error during evaluation loop: {eval_e}", exc_info=True)
        finally:
            if eval_env:
                try:
                    eval_env.close()
                    write_log("   Evaluation environment closed.")
                except Exception as e:
                    write_log(f"   ⚠️ Error closing evaluation environment: {e}")
else:
     write_log("⏩ Skipping evaluation because training was not successful or artifacts are missing.")


# --- Cleanup ---
write_log("🧹 Cleaning up resources...")

if 'train_env' in locals() and train_env:
    try: train_env.close(); write_log("   Train env closed.")
    except Exception as e: write_log(f"   Error closing train env: {e}")

# Close eval_env just in case it wasn't closed in the eval block's finally
if 'eval_env' in locals() and eval_env and ('can_evaluate' not in locals() or not can_evaluate):
     try: eval_env.close(); write_log("   (Unused/Errored) Eval env closed.")
     except Exception as e: write_log(f"   Error closing (unused/errored) eval env: {e}")


# Terminate Java Server Process
if java_process and java_process.poll() is None:
    write_log("   Terminating Java server process...")
    java_process.terminate()
    try:
        java_process.wait(timeout=5)
        write_log("   Java server terminated gracefully.")
    except subprocess.TimeoutExpired:
        write_log("   Java server did not terminate gracefully, killing...")
        java_process.kill()
        try: java_process.wait(timeout=2)
        except: pass
        write_log("   Java server killed.")
    except Exception as e:
        write_log(f"   Error during Java server termination: {e}")
elif java_process:
    write_log("   Java server process already terminated.")
else:
    write_log("   Java server process was not started or already handled.")

# Final check on Wandb run status
if wandb_enabled and wandb.run:
    write_log(f"   Wandb run '{wandb.run.id}' is still marked as active. Finishing.")
    try:
        wandb.finish(exit_code=0 if training_successful else 1)
    except Exception as finish_e:
         write_log(f"   Error during final Wandb finish call: {finish_e}")
elif not wandb_enabled:
     write_log("   Wandb was disabled for this run.")
else:
     write_log("   Wandb run was already finished.")


write_log("🏁 PPO Curriculum Learning script finished.")