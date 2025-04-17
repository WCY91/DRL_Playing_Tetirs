# -*- coding: utf-8 -*-
# ... (Imports remain the same, no collections needed) ...
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
from IPython.display import FileLink, display
import torch
import time
import pygame

# --- Wandb Setup ---
# (Same)
import os
import wandb
from kaggle_secrets import UserSecretsClient
from wandb.integration.sb3 import WandbCallback

# --- Configuration ---
STUDENT_ID = "113598065"
TOTAL_TIMESTEPS = 1000000

# --- Wandb Login ---
# (Same)
try:
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login()
    wandb_enabled = True
except Exception as e:
    print(f"Wandb login failed: {e}. Running without Wandb logging.")
    wandb_enabled = False
    WANDB_API_KEY = None

# --- !!! Config - Focus Solely on Line Clear !!! ---
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": TOTAL_TIMESTEPS,
    "env_id": "TetrisEnv-v1-LineClearFocus", # New ID
    # --- PPO Specific Params (Stabilized) ---
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.15,
    "learning_rate": 1e-4,
    # --- Common Params ---
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Reward Coeffs (MODIFIED - Minimal Penalties) ---
    "reward_line_clear_coeff": 500.0,  # Keep high line clear reward
    "penalty_height_increase_coeff": 0.1, # <<< MODIFIED: Minimal penalty (almost zero)
    "penalty_hole_increase_coeff": 0.1,   # <<< MODIFIED: Minimal penalty (almost zero)
    "penalty_step_coeff": 0.0,         # No survival reward
    "penalty_game_over_coeff": 400.0,    # Keep game over penalty
}

# --- Wandb Init ---
if wandb_enabled:
    run = wandb.init(
        project="tetris-training-line-focus", # New project name
        entity="t113598065-ntut-edu-tw",
        sync_tensorboard=True, monitor_gym=True, save_code=True,
        settings=Settings(init_timeout=180), config=config
    )
    run_id = run.id
else:
    run = None
    run_id = f"local_ppo_linefocus_{int(time.time())}"

log_path = f"/kaggle/working/tetris_train_log_{run_id}.txt"

# --- Helper Functions & Server Start ---
# (Remain unchanged)
def write_log(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"; print(log_message)
    try:
        with open(log_path, "a", encoding="utf-8") as f: f.write(log_message + "\n")
    except Exception as e: print(f"Error writing log: {e}")

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
    if not os.path.exists(jar_file): raise FileNotFoundError(f"JAR file not found: '{jar_file}'")
    java_process = subprocess.Popen(["java", "-jar", jar_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    write_log(f"✅ Java server process started (PID: {java_process.pid})")
    if not wait_for_tetris_server(): raise TimeoutError("Java server did not become available.")
except Exception as e:
    write_log(f"❌ Error starting/waiting for Java server: {e}")
    if java_process and java_process.poll() is None:
        write_log("   Terminating Java process..."); java_process.terminate()
        try: java_process.wait(timeout=2)
        except subprocess.TimeoutExpired: java_process.kill()
    raise

if torch.cuda.is_available(): write_log(f"✅ PyTorch using GPU: {torch.cuda.get_device_name(0)}")
else: write_log("⚠️ PyTorch using CPU.")

# ----------------------------
# 定義 Tetris 環境 (Line Clear Focus)
# ----------------------------
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris (Focus on Line Clear Reward).""" # Updated docstring
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    N_DISCRETE_ACTIONS = 5
    IMG_HEIGHT = 200; IMG_WIDTH = 100; IMG_CHANNELS = 3
    RESIZED_DIM = 84

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), dtype=np.uint8)
        self.server_ip = host_ip; self.server_port = host_port
        self.client_sock = None; self._connect_socket()

        # Reward shaping & stats
        self.current_cumulative_lines = 0
        self.current_height = 0; self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None

        # --- Load Reward Coefficients (Minimal Penalties) ---
        current_config = run.config if run and hasattr(run, 'config') else config
        self.reward_line_clear_coeff = current_config.get("reward_line_clear_coeff", 500.0)
        self.penalty_height_increase_coeff = current_config.get("penalty_height_increase_coeff", 0.1) # Minimal default
        self.penalty_hole_increase_coeff = current_config.get("penalty_hole_increase_coeff", 0.1)   # Minimal default
        self.penalty_step_coeff = current_config.get("penalty_step_coeff", 0.0) # No survival reward
        self.penalty_game_over_coeff = current_config.get("penalty_game_over_coeff", 400.0)

        write_log(f"TetrisEnv initialized (Line Clear Focus).")
        write_log(f"Reward Coeffs: Line={self.reward_line_clear_coeff}, H={self.penalty_height_increase_coeff}, O={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}")

        # Rendering/Error flags
        self.window_surface = None; self.clock = None
        self.is_pygame_initialized = False; self._wandb_log_error_reported = False

    # --- Helper methods (_initialize_pygame, _connect_socket, _send_command, _receive_data, get_tetris_server_response) ---
    # (Remain unchanged)
    def _initialize_pygame(self): #... (same as before)
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                import pygame; pygame.init(); pygame.display.init()
                self.window_surface = pygame.display.set_mode((self.RESIZED_DIM * 4, self.RESIZED_DIM * 4))
                pygame.display.set_caption(f"Tetris Env ({self.server_ip}:{self.server_port})")
                self.clock = pygame.time.Clock(); self.is_pygame_initialized = True; write_log("   Pygame initialized.")
            except Exception as e: write_log(f"⚠️ Error initializing Pygame: {e}"); self.render_mode = None
    def _connect_socket(self): #... (same as before)
        try:
            if self.client_sock: self.client_sock.close()
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0); self.client_sock.connect((self.server_ip, self.server_port))
        except socket.error as e: raise ConnectionError(f"Failed connect @ {self.server_ip}:{self.server_port}: {e}")
    def _send_command(self, command: bytes): #... (same as before)
        if not self.client_sock: raise ConnectionError("Socket disconnected.")
        try: self.client_sock.sendall(command)
        except socket.timeout: raise ConnectionAbortedError("Socket timeout on send")
        except socket.error as e: raise ConnectionAbortedError(f"Socket error on send: {e}")
    def _receive_data(self, size): #... (same as before)
        if not self.client_sock: raise ConnectionError("Socket disconnected.")
        data = b""; self.client_sock.settimeout(10.0)
        try:
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk: raise ConnectionAbortedError("Socket broken (empty chunk)")
                data += chunk
        except socket.timeout: raise ConnectionAbortedError(f"Socket timeout receiving {size} (got {len(data)})")
        except socket.error as e: raise ConnectionAbortedError(f"Socket error on receive: {e}")
        return data
    def get_tetris_server_response(self): #... (same as before)
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
            if np_image is None: write_log("❌ Image decode failed."); return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8)
            self.last_raw_render_frame = resized.copy()
            self.last_observation = observation.copy()
            return is_game_over, cumulative_lines, height, holes, observation
        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e: write_log(f"❌ Connection/Value error: {e}."); return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e: write_log(f"❌ Unexpected error getting response: {e}."); return True, self.current_cumulative_lines, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action):
        # --- Send Action ---
        # (Unchanged)
        command_map = {0: b"move -1\n", 1: b"move 1\n", 2: b"rotate 0\n", 3: b"rotate 1\n", 4: b"drop\n"}
        command = command_map.get(action, b"drop\n")
        if action not in command_map: write_log(f"⚠️ Invalid action: {action}.")
        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            # (Simplified error handling)
            write_log(f"❌ Send failure: {e}")
            reward = -self.penalty_game_over_coeff
            info = {'removed_lines': self.current_cumulative_lines, 'lifetime': self.lifetime, 'final_status': 'send_error'}
            info['terminal_observation'] = self.last_observation.copy()
            if wandb_enabled and run:
                try: wandb.log({"reward/step_total": reward, "reward/step_game_over_penalty": -self.penalty_game_over_coeff})
                except Exception as log_e: pass # Ignore log errors here
            return self.last_observation.copy(), reward, True, True, info

        # --- Get State Update ---
        terminated, next_cumulative_lines, new_height, new_holes, observation = self.get_tetris_server_response()
        lines_cleared_this_step = next_cumulative_lines - self.current_cumulative_lines

        # --- Calculate Reward (Line Clear Focus) ---
        reward = 0.0
        line_clear_reward = 0.0

        # Scaled line clear reward is the main positive signal
        if lines_cleared_this_step == 1: line_clear_reward = 1 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 2: line_clear_reward = 4 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 3: line_clear_reward = 9 * self.reward_line_clear_coeff
        elif lines_cleared_this_step >= 4: line_clear_reward = 25 * self.reward_line_clear_coeff
        reward += line_clear_reward

        # --- Height Penalty (Minimal Coeff) ---
        height_increase = max(0, new_height - self.current_height)
        height_penalty = height_increase * self.penalty_height_increase_coeff
        reward -= height_penalty

        # --- Hole Penalty (Minimal Coeff) ---
        hole_increase = max(0, new_holes - self.current_holes)
        hole_penalty = hole_increase * self.penalty_hole_increase_coeff
        reward -= hole_penalty

        # --- Step Reward/Penalty (None) ---
        step_reward_value = 0.0

        # --- Game Over Penalty ---
        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            write_log(f"💔 GameOver L={next_cumulative_lines} T={self.lifetime+1} | Rews: LC={line_clear_reward:.1f} HP={-height_penalty:.1f} OP={-hole_penalty:.1f} GO={-game_over_penalty:.1f} -> Tot={reward:.1f}")

        # --- Update Internal State ---
        self.current_cumulative_lines = next_cumulative_lines
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        truncated = False
        info = {'removed_lines': self.current_cumulative_lines, 'lifetime': self.lifetime}
        if terminated:
            info['terminal_observation'] = observation.copy()
            info['episode'] = {'r': reward, 'l': self.lifetime, 'lines': self.current_cumulative_lines}

        # --- Wandb Logging ---
        if wandb_enabled and run:
             try:
                 log_data = {
                     "reward/step_total": reward, "reward/step_line_clear": line_clear_reward,
                     "reward/step_height_penalty": -height_penalty, "reward/step_hole_penalty": -hole_penalty,
                     "reward/step_survival_reward": step_reward_value, # Should be 0
                     "reward/step_game_over_penalty": -game_over_penalty if terminated else 0.0,
                     "env/lines_cleared_this_step": lines_cleared_this_step,
                     "env/height_increase": height_increase, "env/hole_increase": hole_increase,
                     "env/current_height": self.current_height, "env/current_holes": self.current_holes,
                     "env/current_lifetime": self.lifetime }
                 wandb.log(log_data)
             except Exception as log_e:
                 if not self._wandb_log_error_reported: print(f"Wandb log error: {log_e}"); self._wandb_log_error_reported = True

        if self.render_mode == "human": self.render()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # (Unchanged reset logic)
        super().reset(seed=seed)
        self._wandb_log_error_reported = False
        for attempt in range(3):
            try:
                self._send_command(b"start\n")
                terminated, server_lines, height, holes, observation = self.get_tetris_server_response()
                if terminated or server_lines != 0:
                    write_log(f"⚠️ Invalid server state on reset {attempt+1}. Retrying...")
                    if attempt < 2:
                        try: self._connect_socket(); time.sleep(0.5 + attempt * 0.5)
                        except ConnectionError as ce: write_log(f"   Reconnect failed: {ce}"); time.sleep(1.0)
                        continue
                    else: raise RuntimeError("Server failed valid reset.")
                self.current_cumulative_lines = 0; self.current_height = height
                self.current_holes = holes; self.lifetime = 0
                self.last_observation = observation.copy()
                info = {'start_height': height, 'start_holes': holes}
                return observation, info
            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError, ValueError) as e:
                 write_log(f"🔌 Connection/Value issue during reset {attempt+1}. Retrying...")
                 if attempt < 2:
                     try: self._connect_socket(); time.sleep(0.5 + attempt * 0.5)
                     except ConnectionError: write_log("   Reconnect failed."); time.sleep(1.0)
                 else: raise RuntimeError(f"Failed reset: {e}")
        raise RuntimeError("Failed reset.")

    def render(self):
        # (Unchanged)
        self._initialize_pygame()
        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None: return
            if self.last_raw_render_frame is not None:
                try:
                    render_frame_rgb = cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0)); pygame.event.pump()
                    pygame.display.flip(); self.clock.tick(self.metadata["render_fps"])
                except Exception as e: write_log(f"⚠️ Pygame render error: {e}")
            else: self.window_surface.fill((0,0,0)); pygame.display.flip()
        elif self.render_mode == "rgb_array":
             if self.last_raw_render_frame is not None: return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
             else: return np.zeros((self.RESIZED_DIM, self.RESIZED_DIM, 3), dtype=np.uint8)

    def close(self):
        # (Unchanged)
        if self.client_sock:
            try: self.client_sock.close(); write_log("   Socket closed.")
            except socket.error as e: write_log(f"   Error closing socket: {e}")
            self.client_sock = None
        if self.is_pygame_initialized:
            try: import pygame; pygame.display.quit(); pygame.quit(); write_log("   Pygame closed.")
            except Exception as e: write_log(f"   Error closing Pygame: {e}")
            self.is_pygame_initialized = False

# --- Environment Setup ---
write_log("✅ Creating env function (Line Clear Focus)...")
def make_env():
    env = TetrisEnv(render_mode=None)
    return env

write_log("✅ Creating Vec Env (Dummy)...")
train_env_base = DummyVecEnv([make_env])

write_log("✅ Wrapping Env (VecFrameStack)...")
n_stack_param = config.get("n_stack", 4)
train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack_param, channels_order="first")

write_log("✅ Wrapping Env (VecNormalize - Rewards Normalized)...")
gamma_param = config.get("gamma", 0.99)
train_env = VecNormalize(train_env_stacked,
                         norm_obs=False,
                         norm_reward=True, # Keep normalization ON
                         gamma=gamma_param)
write_log("   Environment setup complete (NormReward=True).")


# ----------------------------
# PPO Model Setup and Training
# ----------------------------
write_log("🧠 Setting up PPO model...")
model = PPO(
    policy=config["policy_type"],
    env=train_env,
    verbose=1,
    gamma=config["gamma"],
    learning_rate=config["learning_rate"], # Low LR
    n_steps=config["n_steps"],             # 1024
    batch_size=config["batch_size"],       # 64
    n_epochs=config["n_epochs"],           # 10
    gae_lambda=config["gae_lambda"],
    clip_range=config["clip_range"],       # Low clip range (0.1)
    ent_coef=config["ent_coef"],           # High entropy (0.15)
    vf_coef=1.0,                           # High VF coef
    max_grad_norm=0.5,                     # Gradient clipping
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tensorboard_log=f"/kaggle/working/runs/{run_id}" if wandb_enabled else None,
    policy_kwargs=dict(normalize_images=False)
)

write_log(f"   PPO model created. Device: {model.device}")
write_log(f"   Key Config: lr={config['learning_rate']}, ent={config['ent_coef']}, H_pen={config['penalty_height_increase_coeff']}, O_pen={config['penalty_hole_increase_coeff']}, StepPen={config['penalty_step_coeff']}")
write_log(f"   Key PPO Params: vf={model.vf_coef}, grad_norm={model.max_grad_norm}, clip={model.clip_range}")
write_log(f"   VecNormalize: norm_reward={train_env.norm_reward}")


# --- Callbacks ---
if wandb_enabled:
    wandb_callback = WandbCallback(model_save_path=f"/kaggle/working/models/{run_id}", model_save_freq=50000, log="all", verbose=2)
    callback_list = [wandb_callback]
else: callback_list = None

# --- Training ---
write_log(f"🚀 Starting PPO training for {TOTAL_TIMESTEPS} steps (Line Clear Focus)...")
training_successful = False
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, log_interval=1)
    write_log("✅ Training complete!")
    training_successful = True
except Exception as e:
    write_log(f"❌ Training error: {e.__class__.__name__}: {e}", exc_info=True)
    error_save_path = f'/kaggle/working/{STUDENT_ID}_ppo_linefocus_err_{run_id}.zip'
    try:
        model.save(error_save_path)
        write_log(f"   Attempted to save model to {error_save_path}")
        if wandb_enabled and run: wandb.save(error_save_path, base_path="/kaggle/working"); write_log("   Error model uploaded.")
    except Exception as save_e: write_log(f"   ❌ Error saving model after error: {save_e}")
    if run:
        try: run.finish(exit_code=1, quiet=True)
        except Exception as finish_e: write_log(f"   Error finishing Wandb: {finish_e}")

# --- Save Final Model ---
if training_successful:
    stats_path = f"/kaggle/working/vecnormalize_stats_linefocus_{run_id}.pkl"
    final_model_name = f'{STUDENT_ID}_ppo_linefocus_final_{run_id}.zip'
    final_model_path = os.path.join("/kaggle/working", final_model_name)
    try:
        train_env.save(stats_path)
        write_log(f"   VecNormalize stats saved to {stats_path}")
        if wandb_enabled and run: wandb.save(stats_path, base_path="/kaggle/working"); write_log("   VecNormalize stats uploaded.")
        model.save(final_model_path)
        write_log(f"✅ Final model saved: {final_model_path}"); display(FileLink(final_model_path))
        if wandb_enabled and run: wandb.save(final_model_path, base_path="/kaggle/working"); write_log("   Final model uploaded.")
    except Exception as e: write_log(f"❌ Error saving final model/stats: {e}"); training_successful = False

# ----------------------------
# Evaluation
# ----------------------------
if training_successful:
    # (Evaluation code remains the same as previous version)
    write_log("\n🧪 Starting evaluation...")
    try:
        def make_eval_env(): env = TetrisEnv(render_mode="rgb_array"); return env
        eval_env_base = DummyVecEnv([make_eval_env])
        n_stack_eval = config.get("n_stack", 4)
        eval_env_stacked = VecFrameStack(eval_env_base, n_stack=n_stack_eval, channels_order="first")
        eval_env = VecNormalize.load(stats_path, eval_env_stacked)
        eval_env.training = False; eval_env.norm_reward = False
        write_log("   Eval env created."); can_evaluate = True
    except Exception as e: write_log(f"❌ Error creating eval env: {e}"); can_evaluate = False; eval_env = None

    if can_evaluate and eval_env is not None:
        num_eval_episodes = 5; total_rewards, total_lines, total_lifetimes, all_frames = [], [], [], []
        try:
            for i in range(num_eval_episodes):
                obs = eval_env.reset(); done = False; ep_rew, ep_lines, ep_len = 0, 0, 0; frames = []
                while not done:
                    if i == 0: try: frames.append(eval_env.render()) catch Exception: pass
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, infos = eval_env.step(action)
                    ep_rew += reward[0]; done = terminated[0] or truncated[0]; info = infos[0]
                    ep_lines = info.get('removed_lines', ep_lines); ep_len = info.get('lifetime', ep_len)
                write_log(f"   Eval Ep {i+1}: Reward={ep_rew:.2f}, Lines={ep_lines}, Steps={ep_len}")
                total_rewards.append(ep_rew); total_lines.append(ep_lines); total_lifetimes.append(ep_len)
                if i == 0: all_frames = frames
            mean_reward=np.mean(total_rewards); std_reward=np.std(total_rewards); mean_lines=np.mean(total_lines); std_lines=np.std(total_lines); mean_lifetime=np.mean(total_lifetimes); std_lifetime=np.std(total_lifetimes)
            write_log(f"--- Evaluation Results ({num_eval_episodes} eps) ---"); write_log(f"   Avg Reward: {mean_reward:.2f} +/- {std_reward:.2f}"); write_log(f"   Avg Lines: {mean_lines:.2f} +/- {std_lines:.2f}"); write_log(f"   Avg Steps: {mean_lifetime:.2f} +/- {std_lifetime:.2f}")
            if wandb_enabled and run: wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward, "eval/mean_lines": mean_lines, "eval/std_lines": std_lines, "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime})
            if all_frames:
                gif_path = f'/kaggle/working/replay_eval_linefocus_{run_id}.gif'; write_log(f"💾 Saving eval GIF: {gif_path}...")
                try:
                    valid_frames = [f for f in all_frames if isinstance(f, np.ndarray) and f.dtype == np.uint8]
                    if valid_frames: imageio.mimsave(gif_path, valid_frames, fps=10, loop=0); write_log("   GIF saved."); display(FileLink(gif_path));
                    if wandb_enabled and run: wandb.log({"eval/replay": wandb.Video(gif_path, fps=10, format="gif")}); write_log("   GIF uploaded.")
                    else: write_log("   ⚠️ No valid frames.")
                except Exception as e: write_log(f"   ❌ Error saving GIF: {e}")
            else: write_log("   ⚠️ No frames for GIF.")
            csv_path = f'/kaggle/working/tetris_eval_scores_linefocus_{run_id}.csv'
            try:
                with open(csv_path, 'w') as fs:
                    fs.write('episode_id,removed_lines,played_steps,reward\n')
                    for i in range(len(total_lines)): fs.write(f'eval_{i},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                    fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                write_log(f"✅ Eval scores CSV saved: {csv_path}"); display(FileLink(csv_path))
                if wandb_enabled and run: wandb.save(csv_path, base_path="/kaggle/working"); write_log("   CSV uploaded.")
            except Exception as e: write_log(f"   ❌ Error saving CSV: {e}")
        except Exception as eval_e: write_log(f"❌ Error during eval loop: {eval_e}", exc_info=True)
        finally:
             if eval_env: try: eval_env.close(); write_log("   Eval env closed.") except Exception as e: write_log(f"   Error closing eval env: {e}")

# --- Cleanup ---
# (Same cleanup logic)
write_log("🧹 Cleaning up...")
if 'train_env' in locals() and train_env: try: train_env.close(); write_log("   Train env closed.") except Exception as e: write_log(f"   Error closing train env: {e}")
if 'eval_env' in locals() and eval_env and 'can_evaluate' in locals() and not can_evaluate: try: eval_env.close(); write_log("   (Unused) Eval env closed.") except Exception as e: write_log(f"   Error closing (unused) eval env: {e}")
if java_process and java_process.poll() is None:
     write_log("   Terminating Java server...")
     java_process.terminate(); try: java_process.wait(timeout=5); write_log("   Java server terminated.")
     except subprocess.TimeoutExpired: java_process.kill(); write_log("   Java server killed.")
     except Exception as e: write_log(f"    Error terminating Java server: {e}")
elif java_process: write_log("   Java server already terminated.")
else: write_log("   Java server not started or already closed.")
if run:
    is_active = hasattr(run, 'step') and run.step is not None
    if is_active:
        exit_c = 0 if training_successful else 1; status = "successfully" if training_successful else f"(marked as {'failed' if exit_c==1 else 'ok'})"
        try: run.finish(exit_code=exit_c); write_log(f"✨ Wandb run finished {status}.")
        except Exception as finish_e: write_log(f"   Error finishing Wandb run: {finish_e}.")
    else: write_log("✨ Wandb run likely already finished.")
write_log("🏁 PPO LineClearFocus script finished.")