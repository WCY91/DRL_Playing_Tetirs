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
from stable_baselines3.common.vec_env import VecMonitor
# from stable_baselines3.common.callbacks import BaseCallback # Replaced by WandbCallback
import torch
import time
import pygame # Added for rendering in TetrisEnv
from stable_baselines3 import PPO
# --- Wandb Setup ---
import os
import wandb

# Import WandbCallback for SB3 integration
from wandb.integration.sb3 import WandbCallback

# --- Configuration ---
# Set your student ID here for filenames
STUDENT_ID = "113598065"
# Set total training steps
TOTAL_TIMESTEPS = 18500000 # Adjust as needed (e.g., 1M, 2M, 5M)


# --- Wandb Login and Initialization ---
try:
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
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1",
    "gamma": 0.995,
    "n_stack": 4,
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

log_path = f"./kaggle/working/tetris_train_log_{run_id}.txt"

def write_log(message):
    """Appends a message to the log file and prints it."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
            print('writing')
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
        current_config = run.config if run else config # Use global config if no run
        self.episode_total_reward = 0.0
        self.total_lines_this_episode = 0
        self.survival_reward_coeff = current_config.get("survival_reward_coeff", 8.1)
        self.drop_reward_coeff     = current_config.get("drop_reward_coeff",     10)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), # (Channels, Height, Width)
            dtype=np.uint8
        )
        self.eval_a = current_config.get("eval_hole_coeff", 35.0)
        self.eval_b = current_config.get("eval_bumpiness_coeff", 33.5)
        self.eval_c = current_config.get("eval_height_coeff", 36.0)
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

        # --- !!! REWARD SHAPING COEFFICIENTS MODIFIED HERE !!! ---
        # Retrieve from Wandb config if available, otherwise use defaults
        
        self.reward_line_clear_coeff = current_config.get("reward_line_clear_coeff", 80.0)       # INCREASED
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 1.75) # DECREASED
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 1.25)   # DECREASED
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.0)                   # SET TO ZERO
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_coeff", 80.0)     # Kept same for now
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
            stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            observation = np.expand_dims(stretched, axis=0).astype(np.uint8)
            save_dir = "gray_images"
            # os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"gray_tmp.png")
            # cv2.imwrite(os.path.join(save_dir, "gray_tmp.png"), stretched)

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
        # 1. 送指令
        command_map = {
            0: b"move -1\n", 1: b"move 1\n",
            2: b"rotate 0\n", 3: b"rotate 1\n",
            4: b"drop\n"
        }
        command = command_map.get(action)
        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            write_log(f"❌ Ending episode due to send failure in step: {e}")
            terminated = True
            observation = self.last_observation.copy()
            reward = self.penalty_game_over_coeff * -1 # Apply game over penalty directly
            info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'send_error'}
            info['terminal_observation'] = observation # Add terminal observation
            return observation, reward, terminated, False, info # Return immediately
        
        terminated, new_lines, new_h, new_holes, obs = self.get_tetris_server_response()
        lines_cleared = new_lines - self.lines_removed
        self.total_lines_this_episode += lines_cleared
        reward = 0.0

        # (a) drop 獎勵：鼓勵主動落下
        if action == 4:
            reward += 4.0  # 輕微鼓勵 drop 行為

        # (b) 清線獎勵：每清一行 +1000
        lines_cleared = new_lines - self.lines_removed
        if lines_cleared > 0:
            reward += 1000 * lines_cleared

        # (c) 高度變化懲罰（只懲罰上升）
        if new_h > self.current_height:
            reward -= (new_h - self.current_height) * 5.5

        # (d) 洞數變化懲罰 / 獎勵
        delta_holes = new_holes - self.current_holes
        # if delta_holes > 0:
        #     reward -= delta_holes * 2.0  # 增加洞 => 懲罰
        if delta_holes < 0:
            reward += (-delta_holes) * 3.0  # 減少洞 => 獎勵

        # (e) 存活獎勵：活著就給一點點
        if not terminated:
            reward += 0.05
        else:
            reward -= 100.0  # 終局懲罰
        self.lines_removed = new_lines

        # 3. 更新狀態
        self.lines_removed = new_lines
        self.current_height = new_h
        self.current_holes  = new_holes
        self.lifetime      += 1
        self.episode_total_reward += reward
        if terminated and self.lines_removed > 0 :
            write_log(self.lines_removed)

        info = {'removed_lines': new_lines, 'lifetime': self.lifetime}
        # normalized_reward = (reward - self.reward_running_mean) / (np.sqrt(self.reward_running_var) + 1e-8)
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_lines_this_episode = 0
        self.episode_total_reward = 0.0
        # Reset the Wandb error reported flag for the new episode
        self._wandb_log_error_reported = False

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
# ----------------------------
# DQN Model Setup and Training
# ----------------------------
write_log("🧠 設定 DQN 模型...")
from stable_baselines3.common.env_util import make_vec_env
# Let's try A2C by creating 30 environments
train_env = make_vec_env(TetrisEnv, n_envs=40, seed=123456789)
train_env = VecNormalize(train_env, norm_reward = True,norm_obs=False)
print(train_env.num_envs)
# # Define DQN model
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch
from stable_baselines3.common.torch_layers import NatureCNN

class TetrisNatureCNN(NatureCNN):
    """
    在 Nature-CNN 基礎上：
      1. 把 Conv2 的 kernel 從 4×4 -> 5×5，stride 2 (cover 高度梯度)
      2. Conv3 改成 dilation=2，感受野拉大
      3. 加一個 SE attention (channel-wise) 強化關鍵特徵
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)  # ← 先建原始 layers
        self.cnn[0] = nn.Conv2d(1, 32, kernel_size=9, stride=4)
        # ----- 修改 Conv2 -----
        self.cnn[2] = nn.Conv2d(32, 64, kernel_size=7, stride=2)   # index 2 = Conv2

        # ----- 修改 Conv3 (dilated) -----
        # Nature 原 index 4: nn.Conv2d(64, 64, 3, 1)
        self.cnn[4] = nn.Conv2d(64, 64, kernel_size=5, stride=1,
                                padding=2, dilation=2, bias=False)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # ----- 在 Conv3 後插入 SE Block -----
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (N,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 16), nn.SiLU(),
            nn.Linear(16, 64), nn.Sigmoid()
        )

        # 重新計算 flatten size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.forward_cnn(sample).shape[1]
        
        # 重建線性層，保持 features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.SiLU()
        )

    # --------- 只覆寫 CNN 前向，線性層沿用父類 ---------
    def forward_cnn(self, x):
        for i, layer in enumerate(self.cnn):
            if isinstance(layer, nn.ReLU):
                # 替換掉 ReLU 成 SiLU
                x = nn.SiLU()(x)
            else:
                x = layer(x)
            if i == 4:
                x = x + self.res_block(x)
                w = self.se(x).view(x.size(0), -1, 1, 1)
                x = x * w
        return x


import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import NatureCNN


policy_kwargs = dict(
    features_extractor_class = TetrisNatureCNN,
    features_extractor_kwargs = dict(features_dim=256),
)
from torch.nn import functional as F
import torch as th

class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            # Sample from replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Online network selects action
                next_q_values_online = self.policy.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

                next_q_values_target = self.policy.q_net_target(replay_data.next_observations)
                next_q_values = next_q_values_target.gather(1, next_actions)

                # Compute TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Current Q estimate
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Loss (Huber loss preferred)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

model = DoubleDQN(
    policy="CnnPolicy",
    env=train_env,
    verbose=1,
    gamma=0.96, # Use loaded gamma
    learning_rate = 1e-3,
    buffer_size=1_200_000,
    learning_starts=185_000,
    batch_size=256,
    tau=1,
    train_freq=(4, "step"), # Train every step
    gradient_steps=7,
    target_update_interval=19_000,
    exploration_fraction = 0.2, # Use the updated value
    exploration_final_eps=0.03,
    policy_kwargs=policy_kwargs,# As per original code
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
    model.save('113598065_dqn_30env_1M.zip')
except Exception as e:
     write_log(f"❌ 訓練過程中發生錯誤: {e}") # Log exception inf


# ----------------------------
# Evaluation (only if training and saving were successful)
# ----------------------------
if training_successful:
    write_log("\n🧪 開始評估訓練後的模型...")
    import os
    import shutil

    # Test the trained agent
    # using the vecenv
    obs = train_env.reset()
    test_steps = 1000

    replay_folder = './replay'
    if os.path.exists(replay_folder):
        shutil.rmtree(replay_folder)

    n_env = obs.shape[0] # Number of environments. A2C will play all envs
    ep_id = np.zeros(n_env, int)
    ep_steps = np.zeros(n_env, int)
    cum_reward = np.zeros(n_env)
    max_reward = -1e10
    max_game_id = 0
    max_ep_id = 0
    max_rm_lines = 0
    max_lifetime = 0

    for step in range(test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = train_env.step(action)

        if step % 50 == 0:
            print(f"Step {step}")
            print("Action: ", action)
            print("reward=", reward, " done=", done)

        for eID in range(n_env):
            cum_reward[eID] += reward[eID]
            folder = f'{replay_folder}/{eID}/{ep_id[eID]}'
            if not os.path.exists(folder):
                os.makedirs(folder)
            fname = folder + '/' + '{:06d}'.format(ep_steps[eID]) + '.png'
            cv2.imwrite(fname, obs[eID])
            #cv2.imshow("Image" + str(eID), obs[eID])
            #cv2.waitKey(10)
            ep_steps[eID] += 1

            if done[eID]:
                if cum_reward[eID] > max_reward:
                    max_reward = cum_reward[eID]
                    max_game_id = eID
                    max_ep_id = ep_id[eID]
                    max_rm_lines = info[eID]['removed_lines']
                    max_lifetime = info[eID]['lifetime']

                ep_id[eID] += 1
                cum_reward[eID] = 0
                ep_steps[eID] = 0
    best_replay_path = f'{replay_folder}/{max_game_id}/{max_ep_id}'


    print("After playing 30 envs each for ", test_steps, " steps:")
    print(" Max reward=", max_reward, " Best video: " + best_replay_path)
    print(" Removed lines=", max_rm_lines, " lifetime=", max_lifetime)
    with open('tetris_best_score_dqn.csv', 'w') as fs:
        fs.write('id,removed_lines,played_steps\n')
        fs.write(f'0,{max_rm_lines}, {max_lifetime}\n')
        fs.write(f'1,{max_rm_lines}, {max_lifetime}\n')

    import glob
    import imageio

    filenames = sorted(glob.glob(best_replay_path + '/*.png'))

