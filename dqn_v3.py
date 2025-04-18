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
from stable_baselines3.common.env_util import make_vec_env # For consistency
from IPython.display import FileLink, display
import torch
import time
import pygame # Added for rendering in TetrisEnv
import logging # Using logging module for better control

# --- Wandb Setup ---
import wandb
from kaggle_secrets import UserSecretsClient
from wandb.integration.sb3 import WandbCallback

# --- Constants ---
STUDENT_ID = "113598065"
BASE_PROJECT_NAME = "tetris-phased-training-v2" # Updated project name slightly
WANDB_ENTITY = "t113598065-ntut-edu-tw" # Replace with your Wandb entity

# --- File Paths & Server Config ---
JAR_FILE = "TetrisTCPserver_v0.6.jar"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10612
LOG_PATH_BASE = "/kaggle/working/tetris_train_log"
MODEL_SAVE_DIR = "/kaggle/working/models"
STATS_SAVE_DIR = "/kaggle/working" # Directory for VecNormalize stats
TENSORBOARD_LOG_DIR = "/kaggle/working/runs_phase{phase_num}" # Tensorboard log dir structure
REPLAY_GIF_DIR = "/kaggle/working"
EVAL_CSV_DIR = "/kaggle/working"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure model dir exists

# --- Server Commands ---
CMD_MOVE_LEFT = b"move -1\n"
CMD_MOVE_RIGHT = b"move 1\n"
CMD_ROTATE_LEFT = b"rotate 0\n" # Assuming 0 is left/counter-clockwise
CMD_ROTATE_RIGHT = b"rotate 1\n" # Assuming 1 is right/clockwise
CMD_DROP = b"drop\n"
CMD_START = b"start\n"

# --- Phase 1 Configuration ---
config_p1 = {
    "phase": 1,
    "policy_type": "CnnPolicy",
    "total_timesteps": 600000, # <<< Increased from 300k (Feedback: P1 steps insufficient)
    "env_id": "TetrisEnv-v1-P1",
    "n_actions": 4, # <<< No 'drop' action
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "buffer_size": 250000, # Slightly increased buffer
    "learning_starts": 10000,
    "target_update_interval": 5000,
    "exploration_fraction": 0.6, # Keep high exploration fraction
    "exploration_final_eps": 0.1, # End exploration higher
    "batch_size": 64, # <<< Increased from 32 (Feedback: Potential slow FPS). Adjust based on GPU memory.
    "n_stack": 4,
    "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 1000.0,
    "penalty_height_increase_coeff": 0.0,
    "penalty_hole_increase_coeff": 0.0,
    "penalty_step_coeff": 0.0,
    "penalty_game_over_coeff": 50.0
}

# --- Phase 2 Configuration ---
config_p2 = {
    "phase": 2,
    "policy_type": "CnnPolicy",
    "total_timesteps": 700000, # Increased slightly
    "env_id": "TetrisEnv-v1-P2",
    "n_actions": 5, # <<< Add 'drop' action back
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "buffer_size": 350000, # Increase buffer size
    "learning_starts": 10000,
    "target_update_interval": 10000,
    "exploration_fraction": 0.5, # <<< Increased exploration duration (Feedback: P2/3 exploration)
    "exploration_final_eps": 0.08, # <<< End slightly higher (Feedback: P2/3 exploration)
    "batch_size": 64, # <<< Increased from 32
    "n_stack": 4,
    "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 1000.0,
    "penalty_height_increase_coeff": 0.0,
    "penalty_hole_increase_coeff": 0.0,
    "penalty_step_coeff": 0.0,
    "penalty_game_over_coeff": 500.0
}

# --- Phase 3 Configuration ---
config_p3 = {
    "phase": 3,
    "policy_type": "CnnPolicy",
    "total_timesteps": 1000000, # Increased slightly
    "env_id": "TetrisEnv-v1-P3",
    "n_actions": 5,
    "gamma": 0.99,
    "learning_rate": 5e-5,
    "buffer_size": 500000, # Larger buffer for fine-tuning
    "learning_starts": 5000,
    "target_update_interval": 10000,
    "exploration_fraction": 0.2, # <<< Increased exploration duration (Feedback: P2/3 exploration)
    "exploration_final_eps": 0.05, # <<< End slightly higher (Feedback: P2/3 exploration)
    "batch_size": 64, # <<< Increased from 32
    "n_stack": 4,
    "student_id": STUDENT_ID,
    "reward_line_clear_coeff": 1000.0,
    "penalty_height_increase_coeff": 1.0,
    "penalty_hole_increase_coeff": 2.0,
    "penalty_step_coeff": 0.0,
    "penalty_game_over_coeff": 750.0
}

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Setup file handler later when log_file path is known

# --- Wandb Login ---
try:
    user_secrets = UserSecretsClient()
    WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.login()
    wandb_enabled = True
    logging.info("Wandb login successful.")
except Exception as e:
    logging.warning(f"Wandb login failed (running without secrets?): {e}. Running without Wandb logging.")
    wandb_enabled = False
    WANDB_API_KEY = None

# --- Global Variables ---
java_process = None
log_file_handler = None # For adding file logging later

# --- Helper Functions ---
def setup_file_logging(log_file):
    """Sets up logging to a specific file."""
    global log_file_handler
    if log_file_handler:
        logging.getLogger().removeHandler(log_file_handler) # Remove previous handler if exists
    log_file_handler = logging.FileHandler(log_file, encoding='utf-8')
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(log_file_handler)

def start_wandb_run(phase_config):
    """Starts a new Wandb run for a given phase."""
    run_id = f"local_phase{phase_config['phase']}_{int(time.time())}"
    log_file = f"{LOG_PATH_BASE}_phase{phase_config['phase']}_{run_id}.txt"
    setup_file_logging(log_file) # Setup logging for this run

    if wandb_enabled:
        try:
            run = wandb.init(
                project=BASE_PROJECT_NAME,
                entity=WANDB_ENTITY,
                config=phase_config,
                name=f"phase_{phase_config['phase']}_{int(time.time())}",
                job_type=f"train_phase_{phase_config['phase']}",
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                reinit=True
            )
            run_id = run.id # Use the actual wandb run ID
            log_file = f"{LOG_PATH_BASE}_phase{phase_config['phase']}_{run_id}.txt" # Update log file name with wandb ID
            setup_file_logging(log_file) # Re-setup logging with correct file name
            logging.info(f"Wandb Run (Phase {phase_config['phase']}) Initialized: {run.name} ({run_id})")
            return run, run_id, log_file
        except Exception as e:
            logging.error(f"Failed to initialize wandb run: {e}. Continuing without wandb for this phase.")
            logging.info(f"Running Phase {phase_config['phase']} locally (Wandb init failed)")
            return None, run_id, log_file # Return local run_id
    else:
        logging.info(f"Running Phase {phase_config['phase']} locally (Wandb disabled)")
        return None, run_id, log_file

def wait_for_tetris_server(ip=SERVER_IP, port=SERVER_PORT, timeout=60):
    """Waits for the Tetris TCP server to become available."""
    logging.info(f"⏳ Waiting for Tetris TCP server ({ip}:{port})...")
    start_time = time.time()
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_sock:
                test_sock.settimeout(1.0)
                test_sock.connect((ip, port))
            logging.info("✅ Java TCP server ready.")
            return True
        except socket.error:
            if time.time() - start_time > timeout:
                logging.error(f"❌ Timeout waiting for Java TCP server ({timeout}s).")
                return False
            time.sleep(1.0)

def start_java_server():
    global java_process
    if java_process and java_process.poll() is None:
        logging.info("ℹ️ Java server process already running.")
        if not wait_for_tetris_server():
            logging.warning("❌ Existing Java server not responding. Attempting to restart...")
            stop_java_server()
            java_process = None
        else:
            return True

    try:
        logging.info("🚀 Attempting to start Java Tetris server...")
        if not os.path.exists(JAR_FILE):
            logging.error(f"❌ ERROR: JAR file '{JAR_FILE}' not found.")
            raise FileNotFoundError(f"JAR file '{JAR_FILE}' not found.")

        java_process = subprocess.Popen(
            ["java", "-jar", JAR_FILE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logging.info(f"✅ Java server process started (PID: {java_process.pid}).")
        if not wait_for_tetris_server():
            raise TimeoutError("Java server did not become available.")
        return True
    except Exception as e:
        logging.error(f"❌ Error starting or waiting for Java server: {e}", exc_info=True)
        if java_process and java_process.poll() is None:
            java_process.terminate()
            try:
                java_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                java_process.kill()
        java_process = None
        return False

def stop_java_server():
    global java_process
    if java_process and java_process.poll() is None:
        logging.info("🔌 Stopping Java server process...")
        java_process.terminate()
        try:
            java_process.wait(timeout=5)
            logging.info("✅ Java server process terminated.")
        except subprocess.TimeoutExpired:
            logging.warning("⚠️ Java server did not terminate gracefully, killing...")
            java_process.kill()
            logging.info("✅ Java server process killed.")
        java_process = None
    elif java_process:
        logging.info("ℹ️ Java server process already terminated.")
        java_process = None
    else:
        logging.info("ℹ️ No Java server process to stop.")

# --- Modified Tetris Environment ---
class TetrisEnv(gym.Env):
    """Custom Environment for Tetris interacting with a Java TCP server.
       Uses constants, refactored communication, and phase-specific configs.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    IMG_HEIGHT = 200 # Original image size (informative)
    IMG_WIDTH = 100  # Original image size (informative)
    IMG_CHANNELS = 3 # Original image channels (informative)
    RESIZED_DIM = 84 # Dimension after resizing

    def __init__(self, host_ip=SERVER_IP, host_port=SERVER_PORT, render_mode=None, phase_config=None):
        super().__init__()
        if phase_config is None:
            raise ValueError("TetrisEnv requires a 'phase_config' dictionary.")

        self.current_phase_config = phase_config
        self.render_mode = render_mode
        self.phase_num = self.current_phase_config['phase']

        # --- Action Space (Dynamic based on phase) ---
        self.n_actions = self.current_phase_config['n_actions']
        self.action_space = spaces.Discrete(self.n_actions)
        logging.info(f"TetrisEnv Phase {self.phase_num} Initialized with {self.n_actions} actions.")

        # --- Observation Space (Grayscale, Resized) ---
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(1, self.RESIZED_DIM, self.RESIZED_DIM), # (Channels, Height, Width) - SB3 CNN standard
            dtype=np.uint8
        )
        self.server_ip = host_ip
        self.server_port = host_port
        self.client_sock = None
        self._connect_socket() # Connect in init

        # --- State/Reward Variables ---
        self.lines_removed = 0
        self.current_height = 0
        self.current_holes = 0
        self.lifetime = 0
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_raw_render_frame = None # Store last BGR frame for rendering

        # --- Load reward coefficients from phase_config ---
        self.reward_line_clear_coeff = self.current_phase_config.get("reward_line_clear_coeff", 0.0)
        self.penalty_height_increase_coeff = self.current_phase_config.get("penalty_height_increase_coeff", 0.0)
        self.penalty_hole_increase_coeff = self.current_phase_config.get("penalty_hole_increase_coeff", 0.0)
        self.penalty_step_coeff = self.current_phase_config.get("penalty_step_coeff", 0.0)
        self.penalty_game_over_coeff = self.current_phase_config.get("penalty_game_over_coeff", 0.0)
        logging.info(f"  Phase {self.phase_num} Reward Coeffs: Line={self.reward_line_clear_coeff}, H={self.penalty_height_increase_coeff}, O={self.penalty_hole_increase_coeff}, Step={self.penalty_step_coeff}, GO={self.penalty_game_over_coeff}")

        # --- Rendering ---
        self.window_surface = None
        self.clock = None
        self.is_pygame_initialized = False
        self._wandb_log_error_reported = False

    # --- Refactored Socket Communication ---
    def _connect_socket(self):
        """Establishes or re-establishes connection to the game server."""
        try:
            if self.client_sock:
                try:
                    self.client_sock.close()
                except socket.error: pass # Ignore errors on close
            self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_sock.settimeout(10.0) # Timeout for operations
            self.client_sock.connect((self.server_ip, self.server_port))
            # logging.debug(f"Socket connected to {self.server_ip}:{self.server_port}") # Debug level
        except socket.error as e:
            logging.error(f"❌ Socket connection error during connect: {e}")
            raise ConnectionError(f"Failed to connect to Tetris server at {self.server_ip}:{self.server_port}") from e

    def _send_command(self, command: bytes):
        """Sends a command, handles potential errors, tries reconnect."""
        if not self.client_sock:
            logging.warning("Socket not connected. Attempting to reconnect...")
            self._connect_socket() # Try to reconnect

        for attempt in range(2): # Try sending twice with reconnect in between
            try:
                self.client_sock.sendall(command)
                # logging.debug(f"Sent: {command.strip()}")
                return
            except (socket.timeout, socket.error, BrokenPipeError) as e:
                logging.warning(f"Socket error during send (attempt {attempt+1}): {e}. Reconnecting...")
                if attempt == 0:
                    self._connect_socket() # Try reconnecting once
                else:
                    logging.error("❌ Failed to send command after reconnect attempt.")
                    raise ConnectionAbortedError(f"Failed to send command: {e}") from e

    def _receive_data(self, size):
        """Receives exactly size bytes, handles errors, tries reconnect."""
        if not self.client_sock:
             logging.warning("Socket not connected during receive. Attempting to reconnect...")
             self._connect_socket() # Try to reconnect

        data = b""
        try:
            # self.client_sock.settimeout(10.0) # Timeout is set on socket creation
            while len(data) < size:
                chunk = self.client_sock.recv(size - len(data))
                if not chunk:
                    logging.error("❌ Socket connection broken during receive (received empty chunk).")
                    raise ConnectionAbortedError("Socket connection broken")
                data += chunk
            # logging.debug(f"Received {len(data)} bytes.")
            return data
        except (socket.timeout, socket.error, ConnectionAbortedError) as e:
             logging.error(f"❌ Socket error/timeout during receive (expected {size}, got {len(data)}): {e}")
             # Don't automatically reconnect here, let higher level handle episode termination
             raise ConnectionAbortedError(f"Socket error/timeout during receive: {e}") from e

    def _receive_int(self):
        """Receives 4 bytes and converts to int."""
        return int.from_bytes(self._receive_data(4), 'big')

    def _receive_byte(self):
        """Receives 1 byte."""
        return self._receive_data(1)

    def _process_received_image(self, img_png):
        """Decodes PNG bytes, resizes, converts to grayscale observation format."""
        nparr = np.frombuffer(img_png, np.uint8)
        np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Decode as BGR
        if np_image is None:
            logging.error("❌ Failed to decode image from server response.")
            # Return a default/last observation? Or signal error?
            # Let's return the last valid observation to avoid crashing SB3, but log error.
            return self.last_observation.copy()

        # Store the color image for rendering before processing
        self.last_raw_render_frame = cv2.resize(np_image, (self.RESIZED_DIM * 2, self.RESIZED_DIM * 2), interpolation=cv2.INTER_NEAREST) # Upscale slightly for display

        resized = cv2.resize(np_image, (self.RESIZED_DIM, self.RESIZED_DIM), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        observation = np.expand_dims(gray, axis=0).astype(np.uint8) # Add channel dim: (1, H, W)
        self.last_observation = observation.copy() # Update last valid observation
        return observation

    def get_tetris_server_response(self):
        """Gets state update from the Tetris server using refactored methods."""
        try:
            is_game_over_byte = self._receive_byte()
            is_game_over = (is_game_over_byte == b'\x01')

            removed_lines = self._receive_int()
            height = self._receive_int()
            holes = self._receive_int()
            img_size = self._receive_int()

            if img_size <= 0 or img_size > 2000000: # Increased max size slightly
                logging.error(f"❌ Received invalid image size: {img_size}. Aborting receive.")
                # Terminate episode on invalid data from server
                return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

            img_png = self._receive_data(img_size)
            observation = self._process_received_image(img_png)

            return is_game_over, removed_lines, height, holes, observation

        except (ConnectionAbortedError, ConnectionRefusedError, ValueError) as e:
            logging.error(f"❌ Connection/Value error getting server response: {e}. Ending episode.")
            # Return 'terminated' state with last known values
            return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()
        except Exception as e:
            logging.error(f"❌ Unexpected error getting server response: {e}.", exc_info=True)
            # Terminate episode on unexpected error
            return True, self.lines_removed, self.current_height, self.current_holes, self.last_observation.copy()

    # --- Action Mapping ---
    def _get_command_for_action(self, action):
        """Maps action index to server command based on current phase."""
        if self.n_actions == 4: # Phase 1
             # NOTE: Phase 1 (4 actions) assumes the Java server handles auto-dropping
             # the piece after a certain time or number of non-drop commands.
             # If the server requires an explicit 'drop' or 'tick' command to lower
             # the piece, this phase might not train correctly. Server behavior needs verification.
            command_map = {
                0: CMD_MOVE_LEFT, 1: CMD_MOVE_RIGHT,
                2: CMD_ROTATE_LEFT, 3: CMD_ROTATE_RIGHT
            }
            # If action is out of bounds (shouldn't happen), send neutral command
            default_command = CMD_ROTATE_LEFT
        elif self.n_actions == 5: # Phase 2 & 3
            command_map = {
                0: CMD_MOVE_LEFT, 1: CMD_MOVE_RIGHT,
                2: CMD_ROTATE_LEFT, 3: CMD_ROTATE_RIGHT,
                4: CMD_DROP
            }
            default_command = CMD_DROP # Default to drop if action invalid
        else:
            logging.error(f"❌ Invalid number of actions configured: {self.n_actions}")
            raise ValueError(f"Invalid n_actions: {self.n_actions}")

        command = command_map.get(action)
        if command is None:
            logging.warning(f"⚠️ Invalid action received: {action} for {self.n_actions} actions. Sending default: {default_command.strip()}")
            return default_command
        return command

    # --- Gym Methods ---
    def step(self, action):
        # --- Send Action ---
        command = self._get_command_for_action(action)
        try:
            self._send_command(command)
        except (ConnectionAbortedError, ConnectionError) as e:
            logging.error(f"❌ Ending episode due to send failure in step: {e}")
            terminated = True
            observation = self.last_observation.copy()
            # Use the game over penalty directly from config
            reward = self.penalty_game_over_coeff * -1.0 # Make float
            info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'final_status': 'send_error', 'phase': self.phase_num}
            # Add terminal observation for SB3 compatibility if needed by wrappers
            info['terminal_observation'] = observation
            self._log_wandb_step_rewards(reward=reward, line_clear_reward=0, height_penalty=0, hole_penalty=0, step_penalty=0, game_over_penalty=-self.penalty_game_over_coeff, lines_cleared_this_step=0, height_increase=0, hole_increase=0)
            return observation, reward, terminated, False, info # truncated=False

        # --- Get State Update ---
        terminated, new_lines_removed, new_height, new_holes, observation = self.get_tetris_server_response()

        # --- Calculate Reward ---
        reward = 0.0
        lines_cleared_this_step = new_lines_removed - self.lines_removed

        # Multi-line clear reward logic (using current phase coeff)
        line_clear_reward = 0.0
        if lines_cleared_this_step == 1: line_clear_reward = 1 * self.reward_line_clear_coeff
        elif lines_cleared_this_step == 2: line_clear_reward = 4 * self.reward_line_clear_coeff # Quadratic bonus
        elif lines_cleared_this_step == 3: line_clear_reward = 9 * self.reward_line_clear_coeff
        elif lines_cleared_this_step >= 4: line_clear_reward = 16 * self.reward_line_clear_coeff # Adjusted tetris bonus
        reward += line_clear_reward

        # Penalties (using current phase coeffs)
        height_increase = max(0, new_height - self.current_height) # Penalize only increase
        height_penalty = height_increase * self.penalty_height_increase_coeff
        reward -= height_penalty

        hole_increase = max(0, new_holes - self.current_holes) # Penalize only increase
        hole_penalty = hole_increase * self.penalty_hole_increase_coeff
        reward -= hole_penalty

        step_penalty = self.penalty_step_coeff # Survival penalty per step
        reward -= step_penalty

        game_over_penalty = 0.0
        if terminated:
            game_over_penalty = self.penalty_game_over_coeff
            reward -= game_over_penalty
            logging.info(f"💔 Game Over! Phase: {self.phase_num}, Lines: {new_lines_removed}, Lifetime: {self.lifetime + 1}. Final Step Reward: {reward:.2f}")
            # Log detailed breakdown on game over
            logging.debug(f"  GO Breakdown: LC={line_clear_reward:.2f}, HP={-height_penalty:.2f}, OP={-hole_penalty:.2f}, SP={-step_penalty:.2f}, GO={-game_over_penalty:.2f}")

        # --- Update Internal State ---
        self.lines_removed = new_lines_removed
        self.current_height = new_height
        self.current_holes = new_holes
        self.lifetime += 1

        # --- Prepare Return Values ---
        info = {'removed_lines': self.lines_removed, 'lifetime': self.lifetime, 'phase': self.phase_num,
                'current_height': self.current_height, 'current_holes': self.current_holes} # Add more info
        truncated = False # Assuming no truncation logic for now

        if terminated:
            # SB3 expects terminal_observation in info dict when done=True
            info['terminal_observation'] = observation.copy()

        # --- Detailed Wandb Logging ---
        self._log_wandb_step_rewards(reward, line_clear_reward, height_penalty, hole_penalty, step_penalty, game_over_penalty, lines_cleared_this_step, height_increase, hole_increase)

        # Render if requested
        if self.render_mode == "human":
            self.render()
        # Ensure reward is float
        reward = float(reward)

        return observation, reward, terminated, truncated, info

    def _log_wandb_step_rewards(self, reward, line_clear_reward, height_penalty, hole_penalty, step_penalty, game_over_penalty, lines_cleared_this_step, height_increase, hole_increase):
        """Helper function to log step rewards to Wandb if enabled."""
        if wandb_enabled and wandb.run: # Check if wandb run is active
            try:
                log_data = {
                    f"reward_phase{self.phase_num}/step_total": reward,
                    f"reward_phase{self.phase_num}/step_line_clear": line_clear_reward,
                    f"reward_phase{self.phase_num}/step_height_penalty": -height_penalty, # Log penalties as positive values
                    f"reward_phase{self.phase_num}/step_hole_penalty": -hole_penalty,
                    f"reward_phase{self.phase_num}/step_survival_penalty": -step_penalty,
                    f"reward_phase{self.phase_num}/step_game_over_penalty": -game_over_penalty,
                    f"env_phase{self.phase_num}/lines_cleared_this_step": lines_cleared_this_step,
                    f"env_phase{self.phase_num}/height_increase": height_increase,
                    f"env_phase{self.phase_num}/hole_increase": hole_increase,
                    f"env_phase{self.phase_num}/current_height": self.current_height,
                    f"env_phase{self.phase_num}/current_holes": self.current_holes,
                    f"env_phase{self.phase_num}/current_lifetime": self.lifetime
                }
                wandb.log(log_data)
            except Exception as log_e:
                if not self._wandb_log_error_reported:
                    logging.warning(f"Wandb logging error in step (Phase {self.phase_num}): {log_e}")
                    self._wandb_log_error_reported = True # Report only once per reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._wandb_log_error_reported = False # Reset log error flag

        # Optional: Add small delay before reset to ensure server is ready
        # time.sleep(0.1)

        for attempt in range(3): # Retry reset logic
            try:
                # Ensure connection is fresh or re-establish
                if not self.client_sock or self.client_sock.fileno() == -1:
                     logging.info("Socket disconnected before reset, reconnecting.")
                     self._connect_socket()

                self._send_command(CMD_START)
                terminated, lines, height, holes, observation = self.get_tetris_server_response()

                if terminated:
                    logging.warning(f"⚠️ Server reported game over on reset attempt {attempt+1} (Phase {self.phase_num}). Retrying...")
                    if attempt < 2:
                        time.sleep(0.5 + attempt * 0.5)
                        # Force reconnect before retry if server seems stuck
                        self._connect_socket()
                        continue
                    else:
                        logging.error("❌ Server still terminated after multiple reset attempts. Cannot proceed.")
                        raise RuntimeError("Tetris server failed to reset properly.")

                # Reset successful
                self.lines_removed = 0
                self.current_height = height
                self.current_holes = holes
                self.lifetime = 0
                self.last_observation = observation.copy()
                # Reset last raw frame
                if self.last_raw_render_frame is not None:
                    self.last_raw_render_frame.fill(0)

                logging.debug(f"🔄 Environment Reset (Phase {self.phase_num}). Initial state: H={height}, O={holes}")
                info = {'phase': self.phase_num} # Standard info dict for reset
                return observation, info # Return observation and info dict

            except (ConnectionAbortedError, ConnectionError, socket.error, TimeoutError, ConnectionRefusedError, RuntimeError) as e:
                logging.warning(f"🔌 Connection/Runtime issue during reset attempt {attempt+1} (Phase {self.phase_num}) ({e}). Retrying...")
                if attempt < 2:
                    try:
                        time.sleep(0.5 + attempt * 0.5)
                        self._connect_socket() # Attempt reconnect
                    except ConnectionError:
                        logging.error(" Reconnect failed during reset retry.")
                        if attempt == 1: # If second attempt also fails, raise
                            raise RuntimeError(f"Failed to reconnect and reset Tetris server after multiple attempts: {e}") from e
                else: # Final attempt failed
                    logging.error(f"❌ Failed to reset Tetris server after {attempt+1} attempts.")
                    raise RuntimeError(f"Failed to reset Tetris server: {e}") from e

        # Should not be reached if logic is correct
        raise RuntimeError("Failed to reset Tetris server (exited retry loop unexpectedly).")

    # --- Rendering (Simplified) ---
    def _initialize_pygame(self):
        """Initializes Pygame if not already done."""
        if not self.is_pygame_initialized and self.render_mode == "human":
            try:
                pygame.init()
                pygame.display.init()
                # Scale window based on RESIZED_DIM for better viewing
                window_scale = 4
                window_width = self.RESIZED_DIM * window_scale
                window_height = self.RESIZED_DIM * window_scale
                self.window_surface = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption(f"Tetris Env Phase {self.phase_num}")
                self.clock = pygame.time.Clock()
                self.is_pygame_initialized = True
                logging.info("Pygame initialized for rendering.")
            except ImportError:
                logging.warning("Pygame not installed, cannot use 'human' render mode.")
                self.render_mode = None
            except Exception as e:
                logging.error(f"Error initializing Pygame: {e}")
                self.render_mode = None

    def render(self):
        self._initialize_pygame()

        if self.render_mode == "human" and self.is_pygame_initialized:
            if self.window_surface is None:
                logging.warning("⚠️ Render called but Pygame window is not initialized.")
                return

            render_frame = self.last_raw_render_frame
            if render_frame is None:
                # Create a black frame if none exists yet
                 render_frame = np.zeros((self.RESIZED_DIM * 2, self.RESIZED_DIM * 2, 3), dtype=np.uint8)

            try:
                # Convert BGR (from OpenCV) to RGB for Pygame
                render_frame_rgb = cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB)
                # Create a Pygame surface from the numpy array (transpose dimensions for Pygame)
                surf = pygame.surfarray.make_surface(np.transpose(render_frame_rgb, (1, 0, 2)))
                # Scale the surface to fit the window
                surf = pygame.transform.scale(surf, self.window_surface.get_size())
                self.window_surface.blit(surf, (0, 0))
                pygame.event.pump() # Process events
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            except Exception as e:
                logging.warning(f"⚠️ Error during Pygame rendering: {e}")

        elif self.render_mode == "rgb_array":
            # Return the last raw BGR frame, converted to RGB (H, W, C)
            if self.last_raw_render_frame is not None:
                 return cv2.cvtColor(self.last_raw_render_frame, cv2.COLOR_BGR2RGB)
            else:
                 # Return black frame if no observation/render yet
                 return np.zeros((self.RESIZED_DIM * 2, self.RESIZED_DIM * 2, 3), dtype=np.uint8) # Match display size? Or observation size? Let's use display size.


    def close(self):
        logging.debug(f"Closing environment connection (Phase {self.phase_num}).")
        if self.client_sock:
            try:
                self.client_sock.close()
            except socket.error as e:
                logging.warning(f"Error closing socket: {e}")
            self.client_sock = None

        if self.is_pygame_initialized:
            try:
                pygame.display.quit()
                pygame.quit()
                self.is_pygame_initialized = False
                logging.debug("Pygame window closed.")
            except Exception as e:
                logging.warning(f"Error closing Pygame: {e}")

# --- Environment Maker Function ---
def make_env(phase_config):
    """Helper function to create an instance of the Tetris environment."""
    def _init():
        # Pass the phase config to the env constructor
        env = TetrisEnv(phase_config=phase_config)
        return env
    return _init

# --- Main Training Function for a Phase ---
def run_training_phase(phase_config, prev_model_path=None, prev_stats_path=None):
    """Runs the training loop for a single phase."""

    run, run_id, log_file = start_wandb_run(phase_config) # Log file is now set via setup_file_logging
    phase_num = phase_config['phase']
    logging.info(f"========== Starting Training Phase {phase_num} (Run ID: {run_id}) ==========")

    # --- Start Java Server ---
    if not start_java_server():
        logging.error(f"❌ Failed to start/connect Java server for Phase {phase_num}. Aborting phase.")
        if run: run.finish(exit_code=1)
        return None, None # Indicate failure: no model path, no stats path

    # --- Environment Setup ---
    train_env = None # Initialize to None
    try:
        logging.info("Creating Vectorized Environment...")
        # Create the base environment lambda
        env_lambda = make_env(phase_config)
        # Use make_vec_env for consistency (though DummyVecEnv is similar for n_envs=1)
        train_env_base = make_vec_env(env_lambda, n_envs=1, vec_env_cls=DummyVecEnv)

        logging.info("Wrapping Environment (VecFrameStack)...")
        n_stack = phase_config['n_stack']
        train_env_stacked = VecFrameStack(train_env_base, n_stack=n_stack, channels_order="first")

        logging.info("Wrapping Environment (VecNormalize - Rewards Only)...")
        gamma_param = phase_config['gamma']

        # --- VecNormalize Loading/Initialization ---
        # Feedback: Phase 2/3 didn't load previous stats.
        # Decision: For now, we will *save* stats after each phase, but *not load* them
        # into the next training phase by default, because the reward structure changes
        # significantly between phases, which might make old stats less relevant or even harmful.
        # We WILL load the correct stats for the final evaluation.
        load_previous_stats = False # <<< Set this to True if you want to experiment with loading previous stats
        vec_norm_path_to_load = prev_stats_path if load_previous_stats else None

        if vec_norm_path_to_load and os.path.exists(vec_norm_path_to_load):
             logging.info(f"Loading VecNormalize stats from: {vec_norm_path_to_load}")
             train_env = VecNormalize.load(vec_norm_path_to_load, train_env_stacked)
             # Explicitly set parameters for the new phase if needed (e.g., gamma)
             train_env.gamma = gamma_param
             train_env.norm_obs = False # Ensure obs normalization is off
             train_env.norm_reward = True # Ensure reward normalization is on
             train_env.training = True # Ensure it's in training mode
        else:
            if vec_norm_path_to_load: logging.warning(f"Previous stats path specified but not found: {vec_norm_path_to_load}")
            logging.info("Initializing new VecNormalize for reward normalization.")
            train_env = VecNormalize(train_env_stacked, norm_obs=False, norm_reward=True, gamma=gamma_param)

        logging.info("✅ Environment Setup Complete.")

    except Exception as e:
        logging.error(f"❌ Error setting up environment for Phase {phase_num}: {e}", exc_info=True)
        if run: run.finish(exit_code=1)
        if train_env: train_env.close() # Attempt to close env if partially created
        stop_java_server()
        return None, None

    # --- Model Setup ---
    model = None
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        tensorboard_log_path = TENSORBOARD_LOG_DIR.format(phase_num=phase_num) # Use constant
        os.makedirs(tensorboard_log_path, exist_ok=True) # Ensure TB log dir exists

        if prev_model_path and os.path.exists(prev_model_path):
            logging.info(f"🧠 Loading model from previous phase: {prev_model_path}")
            # Load the model, passing the NEW environment
            # SB3 load automatically handles setting the environment on the model.
            # We can override specific parameters like learning_rate if needed.
            model = DQN.load(
                prev_model_path,
                env=train_env, # CRITICAL: Pass the new VecNormalize env
                device=device,
                # Force parameters from the new phase config:
                learning_rate=phase_config['learning_rate'],
                buffer_size=phase_config['buffer_size'],
                learning_starts=phase_config['learning_starts'], # Reset learning starts for new phase?
                target_update_interval=phase_config['target_update_interval'],
                exploration_fraction=phase_config['exploration_fraction'],
                exploration_final_eps=phase_config['exploration_final_eps'],
                batch_size=phase_config['batch_size'],
                tensorboard_log=tensorboard_log_path, # Update tensorboard log path
                # Custom objects might be needed if LR schedule was complex, simple LR usually ok.
                # custom_objects={"learning_rate": phase_config['learning_rate']}
            )
             # Reset exploration schedule explicitly when loading? Often needed.
            model.exploration_schedule = lambda progress: model.exploration_final_eps + (1.0 - model.exploration_final_eps) * (1.0 - progress / model.exploration_fraction) if progress < model.exploration_fraction else model.exploration_final_eps

            logging.info(f"Model loaded. Action space: {model.action_space}")

            # Verify action space (important when loading phase 1 -> phase 2)
            if model.action_space.n != phase_config['n_actions']:
                 # This indicates a problem - DQN policy structure depends on action space size.
                 # Re-initializing the policy head might be needed, or maybe better to create fresh model?
                 # For simplicity, let's raise an error here. A more advanced approach
                 # could involve transferring weights except for the final layer.
                 logging.error(f"❌ FATAL: Loaded model action space ({model.action_space.n}) != Phase config action space ({phase_config['n_actions']})")
                 logging.error(" This requires changing the model's output layer. Consider starting Phase 2 with a new model or implementing weight transfer.")
                 raise ValueError("Action space mismatch requires model architecture change.")

        else:
            if prev_model_path: logging.warning(f"⚠️ Previous model path specified but not found: {prev_model_path}. Creating new model.")
            logging.info(f"🧠 Creating NEW DQN model for Phase {phase_num}...")
            model = DQN(
                policy=phase_config['policy_type'],
                env=train_env,
                verbose=1,
                gamma=phase_config['gamma'],
                learning_rate=phase_config['learning_rate'],
                buffer_size=phase_config['buffer_size'],
                learning_starts=phase_config['learning_starts'],
                batch_size=phase_config['batch_size'],
                tau=1.0, # Default for DQN (sync target network)
                train_freq=(1, "step"), # Train after every env step
                gradient_steps=1, # Perform 1 gradient update per train_freq
                target_update_interval=phase_config['target_update_interval'],
                exploration_fraction=phase_config['exploration_fraction'],
                exploration_final_eps=phase_config['exploration_final_eps'],
                # exploration_initial_eps=1.0, # Default
                policy_kwargs=dict(normalize_images=False), # Assuming image normalization happens elsewhere if needed
                seed=42 + phase_num, # Vary seed per phase slightly
                device=device,
                tensorboard_log=tensorboard_log_path # Use constant path
            )
        logging.info(f"✅ Model setup complete. Using device: {model.device}")

    except Exception as e:
        logging.error(f"❌ Error setting up model for Phase {phase_num}: {e}", exc_info=True)
        if run: run.finish(exit_code=1)
        if train_env: train_env.close()
        stop_java_server()
        return None, None

    # --- Setup Wandb Callback ---
    callback_list = None
    model_save_path_wandb = None
    if run: # If wandb run is active
        model_save_path_wandb = os.path.join(MODEL_SAVE_DIR, f"phase{phase_num}", run_id) # Consistent path
        os.makedirs(model_save_path_wandb, exist_ok=True)
        wandb_callback = WandbCallback(
            gradient_save_freq=10000, # Log gradients periodically
            model_save_path=model_save_path_wandb, # Save intermediate models here
            model_save_freq=50000, # Frequency to save model checkpoints
            log="all", # Log gradients, parameters, etc.
            verbose=2
        )
        callback_list = [wandb_callback]

    # --- Training ---
    logging.info(f"🚀 Starting training for Phase {phase_num} ({phase_config['total_timesteps']} steps)...")
    training_successful = False
    error_save_path = f'/kaggle/working/{STUDENT_ID}_dqn_phase{phase_num}_error_{run_id}.zip'
    final_model_path = None
    stats_path = None # Initialize stats_path

    try:
        model.learn(
            total_timesteps=phase_config['total_timesteps'],
            callback=callback_list,
            log_interval=10, # Log basic stats (like ep_rew_mean) every 10 episodes
            reset_num_timesteps=True # Start timesteps count from 0 for this phase
        )
        logging.info(f"✅ Training Phase {phase_num} complete!")
        training_successful = True

        # --- Save Final Model and Stats (if successful) ---
        # Use run_id in filenames for uniqueness
        final_model_name = f'{STUDENT_ID}_dqn_phase{phase_num}_final_{run_id}.zip'
        final_model_path = os.path.join(STATS_SAVE_DIR, final_model_name) # Save model in main working dir or specific model dir
        logging.info(f"💾 Saving final model to: {final_model_path}")
        model.save(final_model_path)

        # <<< Save VecNormalize stats (Feedback point 3) >>>
        stats_path = os.path.join(STATS_SAVE_DIR, f"vecnormalize_phase{phase_num}_{run_id}.pkl")
        logging.info(f"💾 Saving VecNormalize stats to: {stats_path}")
        train_env.save(stats_path)

        # If using wandb, upload the final model and stats as artifacts
        if run:
            try:
                # Log final model
                model_artifact = wandb.Artifact(f'model-phase{phase_num}-{run_id}', type='model')
                model_artifact.add_file(final_model_path)
                run.log_artifact(model_artifact)
                logging.info("Final model saved as Wandb artifact.")

                # Log VecNormalize stats
                stats_artifact = wandb.Artifact(f'vecnormalize-stats-phase{phase_num}-{run_id}', type='dataset')
                stats_artifact.add_file(stats_path)
                run.log_artifact(stats_artifact)
                logging.info("VecNormalize stats saved as Wandb artifact.")

            except Exception as artifact_e:
                logging.warning(f"⚠️ Failed to save artifacts to Wandb: {artifact_e}")


    except Exception as e:
        logging.error(f"❌ Error during training Phase {phase_num}: {e}", exc_info=True)
        try:
            # Try saving model even on error
            model.save(error_save_path)
            logging.info(f"Attempted to save error model to {error_save_path}")
            if run: run.save(error_save_path) # Upload error model to wandb if possible
        except Exception as save_e:
            logging.error(f"Failed to save error model: {save_e}")
        # Ensure paths are None if training failed before saving
        final_model_path = None
        stats_path = None

    finally:
        # Ensure env is closed after training attempt
        if train_env:
            try:
                train_env.close()
                logging.info("Training environment closed.")
            except Exception as close_e:
                logging.error(f"Error closing training environment: {close_e}")

        # --- Finish Wandb Run ---
        if run:
            exit_c = 0 if training_successful else 1
            run.finish(exit_code=exit_c)
            logging.info(f"Wandb run finished for Phase {phase_num} (Exit Code: {exit_c}).")

    logging.info(f"========== Finished Training Phase {phase_num} ==========")

    # Return paths for the next phase (or None if failed)
    # Return the successfully saved final paths or the error path if that exists
    if training_successful:
        return final_model_path, stats_path
    elif os.path.exists(error_save_path):
        logging.warning(f"Training failed, returning path to error model: {error_save_path}")
        return error_save_path, None # Return error model path, no valid stats
    else:
        return None, None # Failed before any model could be saved

# --- Main Execution Logic ---

final_phase3_model_path = None
final_phase3_stats_path = None
evaluation_log_file = f"{LOG_PATH_BASE}_evaluation_{int(time.time())}.txt" # Define eval log file path

try:
    # --- Phase 1 ---
    phase1_model_path, phase1_stats_path = run_training_phase(config_p1)

    # --- Phase 2 ---
    phase2_model_path, phase2_stats_path = None, None # Initialize
    if phase1_model_path:
        logging.info("Proceeding to Phase 2.")
        phase2_model_path, phase2_stats_path = run_training_phase(
            config_p2,
            prev_model_path=phase1_model_path,
            prev_stats_path=phase1_stats_path # Pass stats path, but loading is controlled inside function
        )
    else:
        logging.error("❌ Phase 1 did not produce a model file. Skipping subsequent phases.")

    # --- Phase 3 ---
    phase3_model_path, phase3_stats_path = None, None # Initialize
    if phase2_model_path:
        logging.info("Proceeding to Phase 3.")
        phase3_model_path, phase3_stats_path = run_training_phase(
            config_p3,
            prev_model_path=phase2_model_path,
            prev_stats_path=phase2_stats_path # Pass stats path
        )
        # Store final paths for evaluation
        final_phase3_model_path = phase3_model_path
        final_phase3_stats_path = phase3_stats_path # <<< This should now be the correct path from P3 save
    else:
        logging.error("❌ Phase 2 did not produce a model file. Skipping Phase 3.")


    # --- Evaluation (After Phase 3) ---
    # Use the specific stats path saved from Phase 3
    if final_phase3_model_path and final_phase3_stats_path and os.path.exists(final_phase3_model_path) and os.path.exists(final_phase3_stats_path):
        setup_file_logging(evaluation_log_file) # Switch logging to evaluation file
        logging.info(f"\n🧪 Starting Evaluation of Final Phase 3 Model: {final_phase3_model_path}")
        logging.info(f"🧪 Using VecNormalize Stats: {final_phase3_stats_path}")

        if not start_java_server():
            logging.error("❌ Failed to start Java server for evaluation. Skipping.")
        else:
            eval_env = None
            try:
                # Create the base evaluation environment using Phase 3 config
                eval_env_lambda = make_env(config_p3)
                eval_env_base = make_vec_env(eval_env_lambda, n_envs=1, vec_env_cls=DummyVecEnv)
                eval_env_stacked = VecFrameStack(eval_env_base, n_stack=config_p3['n_stack'], channels_order="first")

                # <<< Load the Phase 3 VecNormalize statistics (Feedback point 3) >>>
                logging.info(f"Loading VecNormalize stats for evaluation: {final_phase3_stats_path}")
                eval_env = VecNormalize.load(final_phase3_stats_path, eval_env_stacked)
                eval_env.training = False # Set to evaluation mode
                eval_env.norm_reward = False # Evaluate using the raw rewards
                logging.info("Evaluation environment created and stats loaded.")

                # Load the final model
                eval_model = DQN.load(final_phase3_model_path, env=eval_env) # Pass wrapped env
                logging.info(f"Evaluation model loaded: {final_phase3_model_path}")

                # --- Run Evaluation Episodes ---
                num_eval_episodes = 10 # Increase eval episodes for more stable results
                total_rewards = []
                total_lines = []
                total_lifetimes = []
                all_frames = [] # For GIF of the first episode

                for i in range(num_eval_episodes):
                    obs = eval_env.reset()
                    done = False
                    episode_reward = 0.0
                    episode_lines = 0
                    episode_lifetime = 0
                    frames = []
                    logging.info(f"--- Starting Eval Episode {i+1}/{num_eval_episodes} ---")

                    while not done:
                        # Render for GIF (only first episode)
                        if i == 0:
                            try:
                                # Use the env_method to call render on the underlying TetrisEnv
                                render_data_list = eval_env.env_method("render", mode="rgb_array")
                                if render_data_list and render_data_list[0] is not None:
                                    frames.append(render_data_list[0]) # Append the RGB frame
                            except Exception as render_err:
                                logging.warning(f"⚠️ Error getting render frame during eval: {render_err}")

                        action, _ = eval_model.predict(obs, deterministic=True) # Use deterministic actions for eval
                        obs, reward, terminated, infos = eval_env.step(action) # VecEnv returns lists

                        # Process results from the VecEnv (even if n_envs=1)
                        current_reward = reward[0]
                        done = terminated[0] # Use terminated, not truncated
                        info = infos[0]

                        episode_reward += current_reward
                        # Get stats from the info dict if available
                        episode_lines = info.get('removed_lines', episode_lines)
                        episode_lifetime = info.get('lifetime', episode_lifetime)

                        # Log step details for debugging if needed
                        # logging.debug(f"  Eval Step: Action={action[0]}, Reward={current_reward:.2f}, Done={done}")


                    logging.info(f"  Eval Episode {i+1} Finished: Reward={episode_reward:.2f}, Lines={episode_lines}, Steps={episode_lifetime}")
                    total_rewards.append(episode_reward)
                    total_lines.append(episode_lines)
                    total_lifetimes.append(episode_lifetime)
                    if i == 0: all_frames = frames # Save frames from first episode

                # --- Log and Save Eval Results ---
                mean_reward = np.mean(total_rewards)
                std_reward = np.std(total_rewards)
                mean_lines = np.mean(total_lines)
                std_lines = np.std(total_lines)
                mean_lifetime = np.mean(total_lifetimes)
                std_lifetime = np.std(total_lifetimes)

                logging.info("--- Evaluation Results ---")
                logging.info(f" Episodes:   {num_eval_episodes}")
                logging.info(f" Avg Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                logging.info(f" Avg Lines:  {mean_lines:.2f} +/- {std_lines:.2f}")
                logging.info(f" Avg Steps:  {mean_lifetime:.2f} +/- {std_lifetime:.2f}")

                # Log evaluation metrics to the *last active* Wandb run (Phase 3 run)
                # Need to check if wandb.run is still available and corresponds to phase 3
                # This is brittle. Ideally, evaluation is logged to a new run or associated differently.
                # Let's try logging to the current active run if it exists.
                if wandb_enabled and wandb.run: # Check if a run is active
                    try:
                        # Check if the run ID matches the phase 3 run ID pattern if possible
                        # (Requires storing phase 3 run_id globally)
                        # Or just log to whatever run is currently active/resumed.
                        wandb.log({
                            "eval/mean_reward": mean_reward, "eval/std_reward": std_reward,
                            "eval/mean_lines": mean_lines, "eval/std_lines": std_lines,
                            "eval/mean_lifetime": mean_lifetime, "eval/std_lifetime": std_lifetime,
                            "eval/num_episodes": num_eval_episodes
                        })
                        logging.info("Evaluation results logged to Wandb.")
                    except Exception as log_e:
                        logging.warning(f"Wandb logging error during evaluation summary: {log_e}")

                # --- Generate Replay GIF ---
                # Try to extract a unique identifier from the model path
                try:
                    eval_run_id_part = final_phase3_model_path.split('_')[-1].replace('.zip','')
                except:
                     eval_run_id_part = "eval" # Fallback ID

                if all_frames:
                    gif_path = os.path.join(REPLAY_GIF_DIR, f'replay_eval_phase3_{eval_run_id_part}.gif')
                    logging.info(f"💾 Saving evaluation GIF ({len(all_frames)} frames): {gif_path}")
                    try:
                        imageio.mimsave(gif_path, [np.array(frame).astype(np.uint8) for frame in all_frames if frame is not None], fps=15, loop=0)
                        display(FileLink(gif_path))
                        if wandb_enabled and wandb.run:
                            wandb.log({"eval/replay": wandb.Video(gif_path, fps=15, format="gif")})
                            logging.info("Evaluation GIF logged to Wandb.")
                    except Exception as e: logging.error(f"❌ Error saving or logging GIF: {e}")
                else: logging.warning("⚠️ No frames collected for GIF.")

                # --- Save Evaluation Results CSV ---
                csv_filename = f'tetris_evaluation_scores_phase3_{eval_run_id_part}.csv'
                csv_path = os.path.join(EVAL_CSV_DIR, csv_filename)
                try:
                    with open(csv_path, 'w') as fs:
                        fs.write('episode_id,removed_lines,played_steps,reward\n')
                        if total_lines: # Check if list is not empty
                            for i in range(len(total_lines)):
                                fs.write(f'eval_{i+1},{total_lines[i]},{total_lifetimes[i]},{total_rewards[i]:.2f}\n')
                        # Add summary row
                        fs.write(f'eval_avg,{mean_lines:.2f},{mean_lifetime:.2f},{mean_reward:.2f}\n')
                    logging.info(f"✅ Evaluation scores CSV saved: {csv_path}")
                    display(FileLink(csv_path))
                    if wandb_enabled and wandb.run:
                         # Save CSV as artifact or directly
                         csv_artifact = wandb.Artifact(f'evaluation-scores-{eval_run_id_part}', type='results')
                         csv_artifact.add_file(csv_path)
                         run.log_artifact(csv_artifact)
                         # wandb.save(csv_path) # Alternative simpler save
                         logging.info("Evaluation CSV logged to Wandb.")
                except Exception as e: logging.error(f"❌ Error saving or logging CSV: {e}")

            except FileNotFoundError as e:
                logging.error(f"❌ Error: Required file not found for evaluation: {e}. Skipping.")
            except Exception as eval_e:
                logging.error(f"❌ Error during evaluation: {eval_e}", exc_info=True)
            finally:
                if eval_env:
                    eval_env.close()
                    logging.info("Evaluation environment closed.")
    else:
         logging.warning("❌ Skipping evaluation because final model path or stats path is missing or invalid.")
         if not final_phase3_model_path: logging.warning("  Reason: Final model path is missing.")
         if not final_phase3_stats_path: logging.warning("  Reason: Final stats path is missing.")
         if final_phase3_model_path and not os.path.exists(final_phase3_model_path): logging.warning(f"  Reason: Final model file not found at: {final_phase3_model_path}")
         if final_phase3_stats_path and not os.path.exists(final_phase3_stats_path): logging.warning(f"  Reason: Final stats file not found at: {final_phase3_stats_path}")


    # --- Final Cleanup ---
    logging.info("🧹 Final Cleanup...")
    stop_java_server()
    # Ensure last wandb run is finished if still active (e.g., if eval failed mid-way)
    if wandb_enabled and wandb.run:
        try:
            # Check if it's a real run before finishing
             if wandb.run.id and not wandb.run.id.startswith("local"):
                 logging.info(f"Ensuring final Wandb run ({wandb.run.id}) is finished.")
                 wandb.finish(exit_code=0) # Assume success if we got here unless specific error
        except Exception as final_wandb_e:
            logging.warning(f"Error closing final Wandb run: {final_wandb_e}")

    logging.info("🏁 Phased Training Script Execution Finished.")

except Exception as main_e:
    # Catch any top-level errors during phase execution
    logging.critical(f"💥 UNHANDLED EXCEPTION IN MAIN SCRIPT: {main_e}", exc_info=True)
    # Try to clean up robustly
    try:
        stop_java_server()
    except Exception as cleanup_e:
        logging.error(f"Error during final cleanup after main exception: {cleanup_e}")
    if wandb_enabled and wandb.run:
        try:
             if wandb.run.id and not wandb.run.id.startswith("local"):
                wandb.finish(exit_code=1) # Mark run as failed due to error
        except Exception: pass # Ignore errors during final finish on error

finally:
    # Ensure logging handlers are closed properly
    if log_file_handler:
        logging.getLogger().removeHandler(log_file_handler)
        log_file_handler.close()