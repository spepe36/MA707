from stable_baselines3 import PPO
import random
import pygame
from game_objects import Player
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import time


def _get_movement(action):
    """Discrete action mapping"""

    action_mapping = {
        0: (0, -1),  # Up
        1: (0, 1),   # Down
        2: (-1, 0),  # Left
        3: (1, 0),   # Right
        4: (0, 0)    # Stay
    }
    return action_mapping[action]


def calculate_distance(x1, y1, x2, y2):
    """Calculate distance between two points."""

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class GameEnv(gym.Env):
    """Single-agent environment."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, env_width=1200, env_height=900, visual_mode=False):
        """Initialize environment information."""

        super().__init__()

        self.width = env_width
        self.height = env_height
        self.visualize = visual_mode
        self.spawn_interval = 600
        self.current_step = 0
        self.max_steps = 3000  # Simulations can run for max 3000 steps

        # Initialize player
        self.player = Player(env_width // 2, env_height // 2, 30, (255, 0, 0), 4, 1, env_width, env_height, 5, 'player_image.png')

        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = None

        self.enemy = None  # Single enemy
        self._initialize_enemy()

        # Define observation and action spaces
        max_dim = max(env_width, env_height)

        self.observation_space = spaces.Dict({
            "enemy_position": spaces.Box(low=0, high=max_dim, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
            "player_position": spaces.Box(low=0, high=max_dim, shape=(2,), dtype=np.float32)
        })

        n_actions = 5
        self.action_space = spaces.Discrete(n_actions)

    def _initialize_enemy(self):
        """Spawns enemy at one of the sides randomly."""

        enemy_size = 30
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            x = random.randint(0, self.width - enemy_size)
            y = 0
        elif edge == 'bottom':
            x = random.randint(0, self.width - enemy_size)
            y = self.height - enemy_size
        elif edge == 'left':
            x = 0
            y = random.randint(0, self.height - enemy_size)
        else:  # 'right'
            x = self.width - enemy_size
            y = random.randint(0, self.height - enemy_size)

        self.enemy = {
            "x": float(x),
            "y": float(y),
            "velocity": [0.0, 0.0],
            "size": enemy_size,
            "color": "yellow"
        }

    def reset(self, seed=None, options=None):
        """Reset environment to start a new episode."""

        self.current_step = 0
        self.player.x = self.width // 2
        self.player.y = self.height // 2
        self._initialize_enemy()
        return self._get_observation(), {}

    def step(self, action):
        """Function to process what happens in a single frame: movement, rewards, and whether the simulation has ended."""
        self.current_step += 1

        if isinstance(action, np.ndarray):
            action = int(action)

        # Calculate distance before action
        prev_distance = calculate_distance(self.enemy["x"], self.enemy["y"], self.player.x, self.player.y)

        # Player logic
        self.player.avoid_enemies_single_ppo([self.enemy], self.width, self.height)

        # Apply action to the enemy
        movement = _get_movement(action)
        speed = 2.6
        dx, dy = movement[0] * speed, movement[1] * speed
        self.enemy["x"] += dx
        self.enemy["y"] += dy
        self.enemy["velocity"] = [dx, dy]
        self.enemy["x"] = np.clip(self.enemy["x"], 0, self.width)
        self.enemy["y"] = np.clip(self.enemy["y"], 0, self.height)

        # Calculate distance after action
        current_distance = calculate_distance(self.enemy["x"], self.enemy["y"], self.player.x,
                                                    self.player.y)

        # Check collision
        caught = self.check_collision_with_player()

        # Compute reward
        if caught:
            reward = 1.0  # Reward for catching the player
        else:
            # Reward for moving closer, penalty for moving farther
            reward = (prev_distance - current_distance) * 0.1

        # Termination and truncation
        terminated = caught
        truncated = not caught and self.current_step >= self.max_steps

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Function to get an observation state for a given agent."""

        enemy_pos = np.array([self.enemy["x"], self.enemy["y"]], dtype=np.float32)
        enemy_vel = np.array(self.enemy["velocity"], dtype=np.float32)
        player_pos = np.array([self.player.x, self.player.y], dtype=np.float32)

        return {
            "enemy_position": enemy_pos,
            "velocity": enemy_vel,
            "player_position": player_pos
        }

    def render(self):
        """Function to render the game environment."""

        if self.screen is None:
            return

        self.screen.fill((0, 0, 0))
        self.player.draw(self.screen)
        color = (255, 255, 0)  # Yellow in RGB
        pygame.draw.rect(
            self.screen,
            color,
            (self.enemy["x"], self.enemy["y"], self.enemy["size"], self.enemy["size"])
        )
        pygame.display.flip()

    def close(self):
        """Function to close the game environment."""

        if self.visualize:
            pygame.quit()

    def check_collision_with_player(self):
        """Function to check if a player an enemy collide"""

        player_rect = pygame.Rect(self.player.x, self.player.y, self.player.size, self.player.size)
        enemy_rect = pygame.Rect(self.enemy["x"], self.enemy["y"], self.enemy["size"], self.enemy["size"])
        return player_rect.colliderect(enemy_rect)


def train_single_agent_model(model_name, simulation_num):
    """Function to train a model with a single agent."""
    time_steps = 3000*simulation_num
    env = GameEnv(visual_mode=False)

    model_path = f"{model_name}.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
    else:
        print(f"Model '{model_name}' not found. Creating a new model...")
        model = PPO("MultiInputPolicy", env, verbose=1)  # Utilizes a pre-created model from stable_baselines3

    model.learn(total_timesteps=time_steps)

    model.save(model_name)
    print(f"Model '{model_name}' has been saved.")


def visualize_single_agent_model(model_name):
    """Function to visualize a single agent model."""

    env = GameEnv(visual_mode=True)
    model = PPO.load(model_name, env=env)
    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs, reward, terminated, truncated, info)

        env.render()
        time.sleep(0.005)
        done = terminated or truncated

        if done:
            obs, info = env.reset()


# train_single_agent_model('ppo_first_attempt', 1000)

visualize_single_agent_model('models/successful_single_ai')
