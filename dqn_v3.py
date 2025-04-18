# -*- coding: utf-8 -*-
import numpy as np
import socket
import cv2
import subprocess
import os
import shutil
import glob
import imageio
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack, DummyVecEnv
from IPython.display import FileLink, display
import torch
import time
import pygame # Added for rendering in TetrisEnv
# --- Wandb Setup ---
import wandb
from kaggle_secrets import UserSecretsClient
from wandb.integration.sb3 import WandbCallback

# --- Constants ---
STUDENT_ID = "113598065"
BASE_PROJECT_NAME = "tetris-phased-training" # Wandb project name
WANDB_ENTITY = "t113598065-ntut-edu-tw" # Replace with your Wandb entity

# --- Phase 1 Configuration ---
config_p1 = {
    "phase": 1,
    "policy_type": "CnnPolicy",
    "total_timesteps": 300000, # Shorter phase 1
    "env_id": "TetrisEnv-v1-P1",
    "n_actions": 4, # <<< No 'drop' action
    "gamma": 0.99,
    "learning_rate": 1e-4, # Standard LR for starting
    "buffer_size": 200000, # Smaller buffer might be ok for simpler phase
    "learning_starts": 10000,
    "target_update_interval": 5000,
    "exploration_fraction": 0.6, # Explore more initially
    "exploration_final_eps": 0.1, # End exploration higher
    "batch_size": 32,
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Phase 1 Rewards: Focus ONLY on lines ---
    "reward_line_clear_coeff": 1000.0, # High line clear reward
    "penalty_height_increase_coeff": 0.0, # NO penalty
    "penalty_hole_increase_coeff": 0.0, # NO penalty
    "penalty_step_coeff": 0.0, # NO step penalty
    "penalty_game_over_coeff": 50.0 # Minimal game over penalty
}

# --- Phase 2 Configuration ---
config_p2 = {
    "phase": 2,
    "policy_type": "CnnPolicy",
    "total_timesteps": 500000, # Longer phase 2
    "env_id": "TetrisEnv-v1-P2",
    "n_actions": 5, # <<< Add 'drop' action back
    "gamma": 0.99,
    "learning_rate": 1e-4, # Can potentially reduce later if needed (e.g., 5e-5)
    "buffer_size": 300000, # Increase buffer size
    "learning_starts": 10000, # Relearn slightly or keep previous buffer? Start fresh buffer is safer.
    "target_update_interval": 10000,
    "exploration_fraction": 0.3, # Less exploration needed?
    "exploration_final_eps": 0.05, # Lower final exploration
    "batch_size": 32,
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Phase 2 Rewards: Add Drop, Moderate Game Over ---
    "reward_line_clear_coeff": 1000.0, # Keep high
    "penalty_height_increase_coeff": 0.0, # Still NO height penalty
    "penalty_hole_increase_coeff": 0.0, # Still NO hole penalty
    "penalty_step_coeff": 0.0, # NO step penalty
    "penalty_game_over_coeff": 500.0 # <<< Introduce significant Game Over penalty
}

# --- Phase 3 Configuration ---
config_p3 = {
    "phase": 3,
    "policy_type": "CnnPolicy",
    "total_timesteps": 800000, # Longest phase for refinement
    "env_id": "TetrisEnv-v1-P3",
    "n_actions": 5, # Keep 5 actions
    "gamma": 0.99,
    "learning_rate": 5e-5, # <<< Reduce learning rate for fine-tuning
    "buffer_size": 400000, # Larger buffer
    "learning_starts": 5000, # Start learning sooner
    "target_update_interval": 10000,
    "exploration_fraction": 0.1, # Minimal exploration
    "exploration_final_eps": 0.02, # Very low final exploration
    "batch_size": 32,
    "n_stack": 4,
    "student_id": STUDENT_ID,
    # --- Phase 3 Rewards: Add Penalties for Neatness ---
    "reward_line_clear_coeff": 1000.0, # Keep high
    "penalty_height_increase_coeff": 1.0, # <<< Introduce SMALL height penalty
    "penalty_hole_increase_coeff": 2.0, # <<< Introduce SMALL hole penalty (maybe slightly > height)
    "penalty_step_coeff": 0.0, # NO step penalty
    "penalty_game_over_coeff": 750.0 # Can increase slightly more if needed
}


# --- Wandb Login ---
try:
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login()
    wandb_enabled = True
except Exception as e:
    print(f"Wandb login failed (running without secrets?): {e}. Running without Wandb logging.")
    wandb_enabled = False
    WANDB_API_KEY = None

# --- Global Variables ---
java_process = None
log_path_base = "/kaggle/working/tetris_train_log" # Base path, will add phase/run_id

# --- Helper Functions ---
def write_log(message, log_file):
    """Appends a message to the specified log file and prints it."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - {message}"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Error writing to log file {log_file}: {e}")
    print(log_message)

def start_wandb_run(phase_config):
    """Starts a new Wandb run for a given phase."""
    if wandb_enabled:
        run = wandb.init(
            project=BASE_PROJECT_NAME,
            entity=WANDB_ENTITY,
            config=phase_config, # Log phase-specific config
            name=f"phase_{phase_config['phase']}_{int(time.time())}", # Unique run name
            job_type=f"train_phase_{phase_config['phase']}",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            reinit=True # Allow multiple init calls in one script
        )
        run_id = run.id
        log_file = f"{log_path_base}_phase{phase_config['phase']}_{run_id}.txt"
        write_log(f" Wandb Run (Phase {phase_config['phase']}) Initialized: {run.name} ({run_id})", log_file)
        return run, run_id, log_file
    else:
        run_id = f"local_phase{phase_config['phase']}_{int(time.time())}"
        log_file = f"{log_path_base}_phase{phase_config['phase']}_{run_id}.txt"
        write_log(f" Running Phase {phase_config['phase']} locally (Wandb disabled)", log_file)
        return None, run_id, log_file

def wait_for_tetris_server(ip="127.0.0.1", port=10612, timeout=60, log_file="debug_log.txt"):
    """Waits for the Tetris TCP server to become available."""
    write_log(f"⏳ Waiting for Tetris TCP server ({ip}:{port})...", log_file)
    start_time = time.time()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
                test_sock.settimeout(1.0)
                test_sock.connect((ip, port))
            write_log("✅ Java TCP server ready.", log_file)
            return True
        except socket.error as e:
            if time.time() - start_time > timeout:
                write_log(f"❌ Timeout waiting for Java TCP server ({timeout}s).", log_file)
                return False
            time.sleep(1.0)

def start_java_server(log_file):
    global java_process
    if java_process and java_process.poll() is None:
        write_log("ℹ️ Java server process already running.", log_file)
        # Check connection anyway
        if not wait_for_tetris_server(log_file=log_file):
             write_log("❌ Existing Java server not responding. Attempting to restart...", log_file)
             stop_java_server(log_file) # Try stopping it first
             java_process = None # Reset process variable
        else:
             return True # Already running and responding

    # If process is None or terminated, start it
    try:
        write_log("🚀 Attempting to start Java Tetris server...", log_file)
        jar_file = "TetrisTCPserver_v0.6.jar"
        if not os.path.exists(jar_file):
             write_log(f"❌ ERROR: JAR file '{jar_file}' not found.", log_file)
             raise FileNotFoundError(f"JAR file '{jar_file}' not found.")

        java_process = subprocess.Popen(
            ["java", "-jar", jar_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        write_log(f"✅ Java server process started (PID: {java_process.pid}).", log_file)
        if not wait_for_tetris_server(log_file=log_file):
            raise TimeoutError("Java server did not become available.")
        return True
    except Exception as e:
        write_log(f"❌ Error starting or waiting for Java server: {e}", log_file)
        if java_process and java_process.poll() is None:
            java_process.terminate()
            try:
                java_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                java_process.kill()
        java_process = None # Ensure it's None on failure
        return False

def stop_java_server(log_file):
    global java_process
    if java_process and java_process.poll() is None:
        write_log("🔌 Stopping Java server process...", log_file)
        java_process.terminate()
        try:
            java_process.wait(timeout=5)
            write_log("✅ Java server process terminated.", log_file)
        except subprocess.TimeoutExpired:
            write_log("⚠️ Java server did not terminate gracefully, killing...", log_file)
            java_process.kill()
            write_log("✅ Java server process killed.", log_file)
        java_process = None
    elif java_process:
        write_log("ℹ️ Java server process already terminated.", log_file)
        java_process = None
    else:
        write_log("ℹ️ No Java server process to stop.", log_file)

# --- Modified Tetris Environment ---
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris that interacts with a Java TCP server.
       Configurable action space and reward coefficients.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    IMG_HEIGHT = 200
    IMG_WIDTH = 100
    IMG_CHANNELS = 3
    RESIZED_DIM = 84

    # REMOVED N_DISCRETE_ACTIONS from class level

    def __init__(self, host_ip="127.0.0.1", host_port=10612, render_mode=None, phase_config=None, log_file="debug_log.txt"):
        super().__init__()
        if phase_config is None:
            raise ValueError("TetrisEnv requires a 'phase_config' dictionary during initialization.")

        self.log_file = log_file # Store log file path
        self.current_phase_config = phase_config # Store the config
        self.render_mode = render_mode

        # --- Set action space based on config ---
        self.n_actions = self.current_phase_config['n_actions']
        self.action_space = spaces.Discrete(self.n_actions)
        write_log(f" TetrisEnv Phase {self.current_phase_config['phase']} Initialized with {self.n_actions} actions.", self.log_file)

        # --- Observation Space (remains the same) ---
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

        # --- Load reward coefficients from phase_config ---
        self.reward_line_clear_coeff = self.current_phase_config.get("reward_line_clear_coeff", 0.0)
        self.penalty_height_increase_coeff = self.current_phase_config.get("penalty_height_increase_coeff", 0.0)
        self.penalty_hole_increase_coeff = self.current_phase_config.get("penalty_hole_increase_coeff", 0.0)
        self.penalty_step_coeff = self.current_phase_config.get("penalty_step_coeff", 0.0)
        self.penalty_game_over_coeff = self.current_phase_config.get("penalty_game_over_coeff", 0.0)
        write_log(f"  Reward Coeffs: Line={self.reward_line_clear_coeff}, H={self.penalty_height_increase_coeff}, O={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}", self.log_file)

        # For rendering (unchanged)
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False
        self._wandb_log_error_reported = False # Track per-env instance

    def _initialize_pygame(self):
        """Initializes Pygame if not already done."""
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                import pygame
                pygame.init()
                pygame.display.init()
                window_width = self.RESIZED_DIM * 4
                window_height = self.RESIZED_DIM * 4
                self.window_surface = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption(f"Tetris Env Phase {self.current_phase_config['phase']}")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                write_log(" Pygame initialized for rendering.", self.log_file)
            except ImportError:
                write_log(" Pygame not installed, cannot use 'human' render mode.", self.log_file)
                self.render_mode = None
            except Exception as e:
                write_log(f" Error initializing Pygame: {e}", self.log_file)
                self.render_mode = None

    def _connect_socket(self):
        """Establishes connection to the game server."""
        try:
            if self.client_sock:
                self.client_sock.close()
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0)
            self.client_sock.connect((self.server_ip, self.server_port))
            # write_log(f" Socket connected to {self.server_ip}:{self.server_port}", self.log_file) # Less verbose
        except socket.error as e:
            write_log(f"❌ Socket connection error during connect: {e}", self.log_file)
            raise ConnectionError(f"Failed to connect to Tetris server at {self.server_ip}:{self.server_port}") from e

    def _send_command(self, command: bytes):
        """Sends a command to the server, handles potential errors."""
        if not self.client_sock:
             raise ConnectionError("Socket is not connected. Cannot send command.")
        try:
            self.client_sock.sendall(command)
        except socket.timeout:
            write_log("❌ Socket timeout during send.", self.log_file)
            raise ConnectionAbortedError("Socket timeout during send")
        except socket.error as e:
            write_log(f"❌ Socket error during send: {e}", self.log_file)
            raise ConnectionAbortedError(f"Socket error during send: {e}") from e

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
                    write_log("❌ Socket connection broken during receive (received empty chunk).", self.log_file)
                    raise ConnectionAbortedError("Socket connection broken")
                data += chunk
        except socket.timeout:
             write_log(f"❌ Socket timeout during receive (expected {size}, got {len(data)}).", self.log_file)
             raise ConnectionAbortedError("Socket timeout during receive")
        except socket.error as e:
            write_log(f"❌ Socket error during receive: {e}", self.log_file)
            raise ConnectionAbortedError(f"Socket error during receive: {e}") from e
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

            if img_size <= 0 or img_size > 1000000:
                write_log(f"❌ Received invalid image size: {img_size}. Aborting receive.", self.log_file)
                raise ValueError(f"Invalid image size received: {img_size}")

            img_png = self._receive_data(img_size)

            # Decode and preprocess image
            nparr = np.frombuffer(img_png, np.uint8)
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if np_image is None:
                write_log("❌ Failed to decode image from server response.", self.log_file)
                return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

            resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            observation = np.expand_dims(gray, axis=0).astype(np.uint8)

            self.last_raw_render_frame = resized.copy() # Store BGR for render
            self.last_observation = observation.copy()

            return is_game_over, removed_lines, height, holes, observation

        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e:
             write_log(f"❌ Connection/Value error getting server response: {e}. Ending episode.", self.log_file)
             return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            write_log(f"❌ Unexpected error getting server response: {e}. Ending episode.", self.log_file)
            return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

    def step(self, action):
        # --- Send Action ---
        # Map action index to command based on the number of actions available in this phase
        if self.n_actions == 4: # Phase 1: No drop
             command_map = {
                 0: b"move -1\n", 1: b"move 1\n",
                 2: b"rotate 0\n", 3: b"rotate 1\n"
                 # Action 4 (drop) is missing
             }
             # Need to send *something* periodically, let's send a 'no-op' equivalent
             # or maybe the server handles it? Let's assume server ticks down.
             # If action is invalid (shouldn't happen with correct space), maybe log error.
             # The default command needs rethinking here. Maybe rotate 0?
             default_command = b"rotate 0\n" # Safer than drop if action is out of bounds
        elif self.n_actions == 5: # Phase 2 & 3: With drop
            command_map = {
                0: b"move -1\n", 1: b"move 1\n",
                2: b"rotate 0\n", 3: b"rotate 1\n",
                4: b"drop\n"
            }
            default_command = b"drop\n"
        else:
            write_log(f"❌ Invalid number of actions configured: {self.n_actions}", self.log_file)
            raise ValueError(f"Invalid n_actions: {self.n_actions}")

        command = command_map.get(action)
        if command is None:
            write_log(f"⚠️ Invalid action received: {action} for {self.n_actions} actions. Sending default: {default_command.strip()}", self.log_file)
            command = default_command
        else:
            # Send a dummy 'drop' periodically if action space is 4, otherwise piece never lands?
            # Let's test if the Java server auto-drops. If not, this needs rethinking.
            # For now, assume server handles auto-drop.
            pass


        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            write_log(f"❌ Ending episode due to send failure in step: {e}", self.log_file)
            terminated = True
            observation = self.last_observation.copy()
            # Use the game over penalty directly from config
            reward = self.penalty_game_over_coeff * -1
            info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'send_error', 'phase': self.current_phase_config['phase']}
            info['terminal_observation'] = observation

            # Log detailed rewards on send failure termination (using current phase coeffs)
            self._log_wandb_step_rewards(reward=reward, line_clear_reward=0, height_penalty=0, hole_penalty=0, step_penalty=0, game_over_penalty=-self.penalty_game_over_coeff, lines_cleared_this_step=0, height_increase=0, hole_increase=0)

            return observation, reward, terminated, False, info # Return immediately

        # --- Get State Update ---
        terminated, new_lines_removed, new_height, new_holes, observation = self.get_tetris_server_response()

        # --- Calculate Reward ---
        reward = 0.0
        lines_cleared_this_step = new_lines_removed - self.lines_removed

        # --- Multi-line clear reward logic (using current phase coeff) ---
        line_clear_reward = 0.0
        if lines_cleared_this_step == 1:
            line_clear_reward = 1 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 2:
            line_clear_reward = 4 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 3:
            line_clear_reward = 9 * self.reward_line_clear_coeff
        elif lines_cleared_this_step >= 4:
            line_clear_reward = 25 * self.reward_line_clear_coeff
        reward += line_clear_reward

        # --- Penalties (using current phase coeffs) ---
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

        step_penalty = self.penalty_step_coeff
        reward -= step_penalty

        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            write_log(f"💔 Game Over! Phase: {self.current_phase_config['phase']}, Lines: {new_lines_removed}, Lifetime: {self.lifetime + 1}. RewardBreakdown: LC={line_clear_reward:.2f}, HP={-height_penalty:.2f}, OP={-hole_penalty:.2f}, SP={-step_penalty:.2f}, GO={-game_over_penalty:.2f} -> Total={reward:.2f}", self.log_file)

        # --- Update Internal State ---
        self.lines_removed = new_lines_removed
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'phase': self.current_phase_config['phase']}
        truncated = False

        if terminated:
            info['terminal_observation'] = observation.copy()

        # --- Detailed Wandb Logging ---
        self._log_wandb_step_rewards(reward, line_clear_reward, height_penalty, hole_penalty, step_penalty, game_over_penalty, lines_cleared_this_step, height_increase, hole_increase)

        # Optional: Render on step if requested
        if self.render_mode == "human":
             self.render()

        return observation, reward, terminated, truncated, info

    def _log_wandb_step_rewards(self, reward, line_clear_reward, height_penalty, hole_penalty, step_penalty, game_over_penalty, lines_cleared_this_step, height_increase, hole_increase):
        """Helper function to log step rewards to Wandb if enabled."""
        if wandb_enabled and wandb.run: # Check if wandb run is active
             try:
                 log_data = {
                     f"reward_phase{self.current_phase_config['phase']}/step_total": reward,
                     f"reward_phase{self.current_phase_config['phase']}/step_line_clear": line_clear_reward,
                     f"reward_phase{self.current_phase_config['phase']}/step_height_penalty": -height_penalty,
                     f"reward_phase{self.current_phase_config['phase']}/step_hole_penalty": -hole_penalty,
                     f"reward_phase{self.current_phase_config['phase']}/step_survival_penalty": -step_penalty,
                     f"reward_phase{self.current_phase_config['phase']}/step_game_over_penalty": -game_over_penalty,
                     f"env_phase{self.current_phase_config['phase']}/lines_cleared_this_step": lines_cleared_this_step,
                     f"env_phase{self.current_phase_config['phase']}/height_increase": height_increase,
                     f"env_phase{self.current_phase_config['phase']}/hole_increase": hole_increase,
                     f"env_phase{self.current_phase_config['phase']}/current_height": self.current_height,
                     f"env_phase{self.current_phase_config['phase']}/current_holes": self.current_holes,
                     f"env_phase{self.current_phase_config['phase']}/current_lifetime": self.lifetime
                 }
                 # Filter zero reward components (optional)
                 # filtered_log_data = {k: v for k, v in log_data.items() if not (k.startswith("reward") and not k.endswith("game_over_penalty") and v == 0) or k.startswith("env")}
                 wandb.log(log_data) # Log unfiltered data for now
             except Exception as log_e:
                 if not self._wandb_log_error_reported:
                     print(f" Wandb logging error in step (Phase {self.current_phase_config['phase']}): {log_e}")
                     self._wandb_log_error_reported = True # Report only once per env instance per episode reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._wandb_log_error_reported = False # Reset log error flag

        for attempt in range(3):
            try:
                # Ensure connection is fresh for reset
                if not self.client_sock or self.client_sock.fileno() == -1:
                    self._connect_socket()

                self._send_command(b"start\n")
                terminated, lines, height, holes, observation = self.get_tetris_server_response()
                if terminated:
                    write_log(f"⚠️ Server reported game over on reset attempt {attempt+1} (Phase {self.current_phase_config['phase']}). Retrying...", self.log_file)
                    if attempt < 2:
                        time.sleep(0.5 + attempt * 0.5) # Increasing delay
                        self._connect_socket() # Reconnect before retry
                        continue
                    else:
                        write_log("❌ Server still terminated after multiple reset attempts. Cannot proceed.", self.log_file)
                        raise RuntimeError("Tetris server failed to reset properly.")

                # Reset successful
                self.lines_removed = 0
                self.current_height = height
                self.current_holes = holes
                self.lifetime = 0
                self.last_observation = observation.copy()
                # write_log(f"🔄 Environment Reset (Phase {self.current_phase_config['phase']}). Initial state: H={height}, O={holes}", self.log_file)
                info = {'phase': self.current_phase_config['phase']}
                return observation, info

            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError, ConnectionRefusedError) as e:
                write_log(f"🔌 Connection issue during reset attempt {attempt+1} (Phase {self.current_phase_config['phase']}) ({e}). Retrying...", self.log_file)
                if attempt < 2:
                    try:
                        time.sleep(0.5 + attempt * 0.5)
                        self._connect_socket() # Attempt reconnect
                    except ConnectionError:
                        write_log(" Reconnect failed.", self.log_file)
                        if attempt == 1: # If second attempt also fails, raise
                            raise RuntimeError(f"Failed to reconnect and reset Tetris server after multiple attempts: {e}") from e
                else: # Final attempt failed
                    raise RuntimeError(f"Failed to reset Tetris server after multiple attempts: {e}") from e

        # Should not be reached
        raise RuntimeError("Failed to reset Tetris server.")

    def render(self):
        # --- Rendering Logic (mostly unchanged) ---
        self._initialize_pygame() # Ensure pygame is ready if in human mode

        if self.render_mode == "human" and self.is_pygame_initialized:
            import pygame
            if self.window_surface is None:
                write_log("⚠️ Render called but Pygame window is not initialized.", self.log_file)
                return

            if hasattr(self, 'last_raw_render_frame'):
                try:
                    render_frame_rgb = cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.Surface((self.RESIZED_DIM, self.RESIZED_DIM))
                    pygame.surfarray.blit_array(surf, np.transpose(render_frame_rgb, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, self.window_surface.get_size())
                    self.window_surface.blit(surf, (0, 0))
                    pygame.event.pump()
                    pygame.display.flip()
                    self.clock.tick(self.metadata["render_fps"])
                except Exception as e:
                    write_log(f"⚠️ Error during Pygame rendering: {e}", self.log_file)
            else:
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
        # write_log(f" Closing environment connection (Phase {self.current_phase_config['phase']}).", self.log_file)
        if self.client_sock:
            try:
                self.client_sock.close()
            except socket.error as e:
                 write_log(f" Error closing socket: {e}", self.log_file)
            self.client_sock = None

        if self.is_pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.is_pygame_initialized = False
                # write_log(" Pygame window closed.", self.log_file)
            except Exception as e:
                 write_log(f" Error closing Pygame: {e}", self.log_file)


# --- Environment Maker Function ---
def make_env(phase_config, log_file_for_env):
    """Helper function to create an instance of the Tetris environment for a specific phase."""
    # Pass the phase config and log file to the env constructor
    env = TetrisEnv(phase_config=phase_config, log_file=log_file_for_env)
    return env

# --- Main Training Function for a Phase ---
def run_training_phase(phase_config, prev_model_path=None, prev_stats_path=None):
    """Runs the training loop for a single phase."""

    run, run_id, log_file = start_wandb_run(phase_config)
    phase_num = phase_config['phase']
    write_log(f"========== Starting Training Phase {phase_num} ==========", log_file)

    # --- Start Java Server (ensure it's running for this phase) ---
    if not start_java_server(log_file):
        write_log(f"❌ Failed to start/connect Java server for Phase {phase_num}. Aborting phase.", log_file)
        if run: run.finish(exit_code=1)
        return None, None # Indicate failure

    # --- Environment Setup ---
    write_log(" Creating Vectorized Environment...", log_file)
    try:
        # Create the base environment lambda, passing the log file path
        env_lambda = lambda: make_env(phase_config, log_file)
        train_env_base = DummyVecEnv([env_lambda])

        write_log(" Wrapping Environment (VecFrameStack)...", log_file)
        n_stack = phase_config['n_stack']
        train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack, channels_order="first")

        write_log(" Wrapping Environment (VecNormalize - Rewards Only)...", log_file)
        gamma_param = phase_config['gamma']
        # Important: Always re-initialize VecNormalize for the phase unless loading PREVIOUS phase's stats
        # For simplicity and robustness against changing reward scales, let's re-initialize reward normalization per phase.
        # If prev_stats_path IS provided, we might load it, but be cautious.
        # Let's stick to re-initializing reward norm for now.
        # if prev_stats_path and os.path.exists(prev_stats_path):
        #     write_log(f" Loading VecNormalize stats from: {prev_stats_path}", log_file)
        #     train_env = VecNormalize.load(prev_stats_path, train_env_stacked)
        #     train_env.norm_obs = False # Ensure obs normalization is off
        #     train_env.norm_reward = True # Ensure reward normalization is on
        # else:
        #     if prev_stats_path: write_log(f" Warning: Previous stats path specified but not found: {prev_stats_path}", log_file)
        write_log(" Initializing new VecNormalize for reward normalization.", log_file)
        train_env = VecNormalize(train_env_stacked, norm_obs=False, norm_reward=True, gamma=gamma_param)

        write_log("✅ Environment Setup Complete.", log_file)

    except Exception as e:
        write_log(f"❌ Error setting up environment for Phase {phase_num}: {e}", log_file, exc_info=True)
        if run: run.finish(exit_code=1)
        stop_java_server(log_file) # Stop server if env setup failed
        return None, None

    # --- Model Setup ---
    model = None
    try:
        if prev_model_path and os.path.exists(prev_model_path):
            write_log(f"🧠 Loading model from previous phase: {prev_model_path}", log_file)
            # Load the model, passing the NEW environment and potentially updated learning rate/params
            model = DQN.load(
                prev_model_path,
                env=train_env, # CRITICAL: Pass the new environment for the current phase
                device="cuda" if torch.cuda.is_available() else "cpu",
                # Set learning rate from the current phase config
                learning_rate=phase_config['learning_rate'],
                # You might need to adjust other parameters if they differ significantly,
                # but SB3 load handles many things. Check SB3 docs if issues arise.
                # For example, buffer size is often part of the loaded model state.
                # To be safe, explicitly set buffer_size if needed, though SB3 might override from saved state.
                buffer_size=phase_config['buffer_size'], # Set buffer size explicitly
                learning_starts=phase_config['learning_starts'], # Reset learning starts?
                target_update_interval=phase_config['target_update_interval'],
                exploration_fraction=phase_config['exploration_fraction'],
                exploration_final_eps=phase_config['exploration_final_eps'],
                # Ensure policy_kwargs match if necessary, though usually loaded.
                # policy_kwargs=dict(normalize_images=False), # Likely loaded, but set if needed
                custom_objects={"learning_rate": phase_config['learning_rate'], "lr_schedule": lambda _: phase_config['learning_rate']} # Ensure LR is reset
            )
            # Reset exploration schedule if loading? DQN load might handle this. Test.
            # model.exploration_rate = model.exploration_initial_eps # Reset if needed
            write_log(f" Model loaded. Action space: {model.action_space}", log_file)
            # Verify action space matches
            if model.action_space.n != phase_config['n_actions']:
                 write_log(f"❌ ERROR: Loaded model action space ({model.action_space.n}) != Phase config action space ({phase_config['n_actions']})", log_file)
                 raise ValueError("Action space mismatch between loaded model and current phase environment.")
        else:
            if prev_model_path: write_log(f"⚠️ Previous model path specified but not found: {prev_model_path}. Creating new model.", log_file)
            write_log("🧠 Creating NEW DQN model for Phase {phase_num}...", log_file)
            model = DQN(
                policy=phase_config['policy_type'],
                env=train_env,
                verbose=1,
                gamma=phase_config['gamma'],
                learning_rate=phase_config['learning_rate'],
                buffer_size=phase_config['buffer_size'],
                learning_starts=phase_config['learning_starts'],
                batch_size=phase_config['batch_size'],
                tau=1.0, # Default for DQN
                train_freq=(1, "step"),
                gradient_steps=1,
                target_update_interval=phase_config['target_update_interval'],
                exploration_fraction=phase_config['exploration_fraction'],
                exploration_final_eps=phase_config['exploration_final_eps'],
                policy_kwargs=dict(normalize_images=False), # Assume consistency
                seed=42, # Use consistent seed? Or vary per phase?
                device="cuda" if torch.cuda.is_available() else "cpu",
                tensorboard_log=f"/kaggle/working/runs_phase{phase_num}/{run_id}" if run else None
            )
        write_log(f"✅ Model setup complete. Device: {model.device}", log_file)
        # write_log(f" Model Parameters: {model.get_parameters()['policy']}", log_file) # Can be verbose

    except Exception as e:
        write_log(f"❌ Error setting up model for Phase {phase_num}: {e}", log_file, exc_info=True)
        if run: run.finish(exit_code=1)
        if 'train_env' in locals(): train_env.close()
        stop_java_server(log_file)
        return None, None

    # --- Setup Wandb Callback ---
    callback_list = None
    if run: # If wandb run is active
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"/kaggle/working/models/phase{phase_num}/{run_id}",
            model_save_freq=50000,
            log="all",
            verbose=2
        )
        callback_list = [wandb_callback]

    # --- Training ---
    write_log(f"🚀 Starting training for Phase {phase_num} ({phase_config['total_timesteps']} steps)...", log_file)
    training_successful = False
    error_save_path = f'/kaggle/working/{STUDENT_ID}_dqn_phase{phase_num}_error_{run_id}.zip'
    try:
        model.learn(
            total_timesteps=phase_config['total_timesteps'],
            callback=callback_list,
            log_interval=10, # Log basic stats every 10 episodes
            # Reset num_timesteps=False ensures continued timestep count if desired,
            # but for distinct phases, starting from 0 is usually clearer.
            reset_num_timesteps=True # Start timesteps count from 0 for this phase
        )
        write_log(f"✅ Training Phase {phase_num} complete!", log_file)
        training_successful = True
    except Exception as e:
         write_log(f"❌ Error during training Phase {phase_num}: {e}", log_file, exc_info=True)
         try:
             model.save(error_save_path)
             write_log(f" Attempted to save error model to {error_save_path}", log_file)
             if run: run.save(error_save_path)
         except Exception as save_e:
              write_log(f"  Failed to save error model: {save_e}", log_file)
    finally:
        # Ensure env is closed after training attempt
        if 'train_env' in locals():
             try:
                 train_env.close() # Close wrapped env first
                 write_log(" Training environment closed.", log_file)
             except Exception as close_e:
                 write_log(f" Error closing training environment: {close_e}", log_file)


    # --- Save Final Model and Stats (if successful) ---
    final_model_path = None
    stats_path = None
    if training_successful:
        stats_path = f"/kaggle/working/vecnormalize_phase{phase_num}_{run_id}.pkl"
        final_model_name = f'{STUDENT_ID}_dqn_phase{phase_num}_final_{run_id}.zip'
        final_model_path = os.path.join("/kaggle/working", final_model_name)


    # --- Finish Wandb Run ---
    if run:
        exit_c = 0 if training_successful else 1
        run.finish(exit_code=exit_c)
        write_log(f" Wandb run finished for Phase {phase_num} (Exit Code: {exit_c}).", log_file)

    write_log(f"========== Finished Training Phase {phase_num} ==========", log_file)

    # Return paths for the next phase (or None if failed)
    if training_successful:
        return final_model_path, stats_path
    else:
        # Return the error path if it exists, otherwise None
        if os.path.exists(error_save_path):
            return error_save_path, None # Return error model path, no valid stats
        else:
            return None, None


# --- Main Execution Logic ---

final_phase3_model_path = None
final_phase3_stats_path = None

try:
    # --- Phase 1 ---
    phase1_model_path, phase1_stats_path = run_training_phase(config_p1)

    # --- Phase 2 ---
    if phase1_model_path: # Proceed only if phase 1 saved a model (success or error)
        # If phase 1 failed, phase1_model_path might be the error model.
        # Decide if you want to continue from an error model or stop.
        # For now, assume we continue even from an error model if it was saved.
        phase2_model_path, phase2_stats_path = run_training_phase(
            config_p2,
            prev_model_path=phase1_model_path # Use the output model from P1
            # prev_stats_path=phase1_stats_path # Not using stats loading for now
        )
    else:
        print(f"❌ Phase 1 did not produce a model file. Skipping subsequent phases.")
        phase2_model_path = None

    # --- Phase 3 ---
    if phase2_model_path:
        phase3_model_path, phase3_stats_path = run_training_phase(
            config_p3,
            prev_model_path=phase2_model_path # Use the output model from P2
            # prev_stats_path=phase2_stats_path # Not using stats loading
        )
        # Store final paths for evaluation
        final_phase3_model_path = phase3_model_path
        final_phase3_stats_path = phase3_stats_path # Need to save P3 stats
    else:
        print(f"❌ Phase 2 did not produce a model file. Skipping Phase 3.")


    # --- Evaluation (After Phase 3) ---
    if final_phase3_model_path and final_phase3_stats_path:
        eval_log_file = f"{log_path_base}_evaluation_{int(time.time())}.txt"
        write_log("\n🧪 Starting Evaluation of Final Phase 3 Model...", eval_log_file)

        if not start_java_server(eval_log_file): # Ensure server is running for eval
             write_log("❌ Failed to start Java server for evaluation. Skipping.", eval_log_file)
        else:
             eval_env = None
             try:
                 # Use Phase 3 config for evaluation environment consistency
                 eval_env_lambda = lambda: make_env(config_p3, eval_log_file)
                 eval_env_base = DummyVecEnv([eval_env_lambda])

                 eval_env_stacked = VecFrameStack(eval_env_base, n_stack=config_p3['n_stack'], channels_order="first")

                 # Load the Phase 3 VecNormalize statistics
                 write_log(f" Loading VecNormalize stats for evaluation: {final_phase3_stats_path}", eval_log_file)
                 eval_env = VecNormalize.load(final_phase3_stats_path, eval_env_stacked)
                 eval_env.training = False
                 eval_env.norm_reward = False # Evaluate actual rewards
                 write_log(" Evaluation environment created.", eval_log_file)

                 # Load the final model
                 eval_model = DQN.load(final_phase3_model_path, env=eval_env)
                 write_log(f" Evaluation model loaded: {final_phase3_model_path}", eval_log_file)

                 # --- Run Evaluation Episodes ---
                 num_eval_episodes = 5
                 total_rewards = []
                 total_lines = []
                 total_lifetimes = []
                 all_frames = []

                 for i in range(num_eval_episodes):
                     obs = eval_env.reset()
                     done = False
                     episode_reward = 0
                     episode_lines = 0
                     episode_lifetime = 0
                     frames = []
                     last_info = {}

                     while not done:
                         # Render for GIF (optional, only first episode)
                         if i == 0:
                              try:
                                  # Access underlying env - careful with DummyVecEnv structure
                                  base_env = eval_env.envs[0] # DummyVecEnv stores envs in a list
                                  # Need to unwrap further if Monitor or other wrappers are present.
                                  # Assuming VecNormalize -> VecFrameStack -> DummyVecEnv -> TetrisEnv
                                  # This path might be complex. A simpler way:
                                  # Directly call the base env's render?
                                  # Let's try getting the attribute directly from VecEnv
                                  raw_frame = eval_env.render(mode="rgb_array") # VecEnvs might not support this directly
                                  # Fallback: Get attr 'last_raw_render_frame'
                                  # render_frames = eval_env.get_attr('last_raw_render_frame') # Gets list
                                  # if render_frames and render_frames[0] is not None:
                                  #    raw_frame = cv2.cvtColor(render_frames[0], cv2.COLOR_BGR2RGB)
                                  #    frames.append(raw_frame)

                                  # Let's use the env's own render method accessed via getattr
                                  render_data_list = eval_env.env_method("render", mode="rgb_array")
                                  if render_data_list and render_data_list[0] is not None:
                                      frames.append(render_data_list[0])

                              except Exception as render_err:
                                  write_log(f"⚠️ Error getting render frame during eval: {render_err}", eval_log_file)

                         action, _ = eval_model.predict(obs, deterministic=True)
                         obs, reward, done, infos = eval_env.step(action)

                         episode_reward += reward[0]
                         last_info = infos[0]
                         episode_lines = last_info.get('removed_lines', episode_lines)
                         episode_lifetime = last_info.get('lifetime', episode_lifetime)
                         done = done[0]

                     write_log(f"  Eval Episode {i+1}: Reward={episode_reward:.2f}, Lines={episode_lines}, Steps={episode_lifetime}", eval_log_file)
                     total_rewards.append(episode_reward)
                     total_lines.append(episode_lines)
                     total_lifetimes.append(episode_lifetime)
                     if i == 0: all_frames = frames

                 # --- Log and Save Eval Results ---
                 mean_reward = np.mean(total_rewards)
                 std_reward = np.std(total_rewards)
                 mean_lines = np.mean(total_lines)
                 std_lines = np.std(total_lines)
                 mean_lifetime = np.mean(total_lifetimes)
                 std_lifetime = np.std(total_lifetimes)

                 write_log("--- Evaluation Results ---", eval_log_file)
                 write_log(f"  Avg Reward: {mean_reward:.2f} +/- {std_reward:.2f}", eval_log_file)
                 write_log(f"  Avg Lines:  {mean_lines:.2f} +/- {std_lines:.2f}", eval_log_file)
                 write_log(f"  Avg Steps:  {mean_lifetime:.2f} +/- {std_lifetime:.2f}", eval_log_file)

                 # Log evaluation metrics to the *last active* Wandb run (Phase 3 run)
                 # Need to re-init wandb or find the run object if script restarted?
                 # Assuming the run object from phase 3 might still be available if run sequentially.
                 # For robustness, maybe save results locally and upload later.
                 # Let's try logging if wandb.run is available.
                 if wandb_enabled and wandb.run and wandb.run.id.startswith("local_phase3") == False : # Check if a real run is active
                     try:
                        wandb.log({
                             "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                             "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                             "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime,
                         })
                     except Exception as log_e:
                         write_log(f" Wandb logging error during evaluation summary: {log_e}", eval_log_file)


                 # --- Generate Replay GIF ---
                 eval_run_id_part = final_phase3_model_path.split('_')[-1].replace('.zip','') # Extract run id part
                 if all_frames:
                     gif_path = f'/kaggle/working/replay_eval_phase3_{eval_run_id_part}.gif'
                     write_log(f"💾 Saving evaluation GIF: {gif_path}", eval_log_file)
                     try:
                         imageio.mimsave(gif_path, [np.array(frame).astype(np.uint8) for frame in all_frames if frame is not None], fps=15, loop=0)
                         display(FileLink(gif_path))
                         if wandb_enabled and wandb.run and wandb.run.id.startswith("local_phase3") == False:
                             wandb.log({"eval/replay": wandb.Video(gif_path, fps=15, format="gif")})
                     except Exception as e: write_log(f"  ❌ Error saving GIF: {e}", eval_log_file)
                 else: write_log("  ⚠️ No frames collected for GIF.", eval_log_file)

                 # --- Save Evaluation Results CSV ---
                 csv_filename = f'tetris_evaluation_scores_phase3_{eval_run_id_part}.csv'
                 csv_path = os.path.join("/kaggle/working", csv_filename)
                 try:
                     with open(csv_path, 'w') as fs:
                         fs.write('episode_id,removed_lines,played_steps,reward\n')
                         if total_lines:
                              for i in range(len(total_lines)):
                                  fs.write(f'eval_{i},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                         fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                     write_log(f"✅ Evaluation scores CSV saved: {csv_path}", eval_log_file)
                     display(FileLink(csv_path))
                     if wandb_enabled and wandb.run and wandb.run.id.startswith("local_phase3") == False:
                        wandb.save(csv_path)
                 except Exception as e: write_log(f"  ❌ Error saving CSV: {e}", eval_log_file)

             except FileNotFoundError as e:
                 write_log(f"❌ Error: Required file not found for evaluation: {e}. Skipping.", eval_log_file)
             except Exception as eval_e:
                 write_log(f"❌ Error during evaluation: {eval_e}", eval_log_file, exc_info=True)
             finally:
                 if eval_env:
                     eval_env.close()
                     write_log(" Evaluation environment closed.", eval_log_file)

    # --- Final Cleanup ---
    write_log("🧹 Final Cleanup...", "cleanup_log.txt") # Use a generic log for final cleanup
    stop_java_server("cleanup_log.txt")
    # Ensure last wandb run is finished if still active (e.g., if eval failed)
    if wandb_enabled and wandb.run:
        try:
            if wandb.run.id.startswith("local") == False: # Don't try to finish local runs
                 wandb.finish(exit_code=0) # Assume success if we got here unless specific error
                 write_log(" Final Wandb run closed.", "cleanup_log.txt")
        except Exception as final_wandb_e:
             write_log(f" Error closing final Wandb run: {final_wandb_e}", "cleanup_log.txt")

    write_log("🏁 Phased Training Script Execution Finished.", "cleanup_log.txt")

except Exception as main_e:
    # Catch any top-level errors
    print(f"💥 UNHANDLED EXCEPTION IN MAIN SCRIPT: {main_e}", exc_info=True)
    # Try to clean up
    stop_java_server("error_cleanup_log.txt")
    if wandb_enabled and wandb.run:
        try:
             if wandb.run.id.startswith("local") == False:
                 wandb.finish(exit_code=1) # Mark as failed
        except Exception: pass # Ignore errors during final finish on error