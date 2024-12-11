import functools
import pygame
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
from pettingzoo import ParallelEnv
from game_objects import Player
import random

MAX_STEPS = 5000
ACTION_MAP = {
        0: (0, -2.6),  # Up
        1: (0, 2.6),  # Down
        2: (-2.6, 0),  # Left
        3: (2.6, 0),  # Right
        4: (0, 0)  # Stay
    }


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class NeuralDash(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, env_width=700, env_height=700, number_of_enemies=2, render_mode=None):

        # Game Information
        self.state = None
        self.total_enemies = number_of_enemies
        self.width = env_width
        self.height = env_height
        self.possible_agents = ["enemy_" + str(r) for r in range(number_of_enemies)]
        self.frames = 0
        self.last_actions = {}
        self.consecutive_close_steps = {}

        # Player and Enemy Initialization
        self.player = Player(env_width // 2, env_height // 2, 30, (255, 0, 0), 4, 1, env_width, env_height, 5, 'player_image.png')
        self._initialize_enemies()

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = None

    def _initialize_enemies(self):
        self.agent_data = {}  # Initialize as a dictionary to store enemy data

        for name in self.possible_agents:
            enemy_size = 30
            edge = random.choice(['top', 'bottom', 'left', 'right'])

            # Determine spawn position based on the edge
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

            # Add the enemy's data to the dictionary
            self.agent_data[name] = {
                "x": float(x),
                "y": float(y),
                "size": enemy_size,
                "color": "yellow"
            }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # The maximum and minimum positions in the environment
        max_position = np.array([self.width, self.height])
        min_position = -max_position

        # Observation space for relative position to the player (2 values: x, y)
        relative_player_low = -max_position
        relative_player_high = max_position

        # Observation space for relative positions to other agents (N agents x 2 values each)
        relative_agents_low = np.tile(min_position, (self.total_enemies - 1, 1))  # Exclude self
        relative_agents_high = np.tile(max_position, (self.total_enemies - 1, 1))

        # Observation space for distances (1 scalar value for player, N-1 scalars for other agents)
        distance_low = np.array([0])  # Minimum distance is 0
        distance_high = np.array([np.linalg.norm(max_position)])  # Max possible distance

        # Combine everything into a single observation space
        return Dict({
            "relative_player_position": Box(
                low=relative_player_low, high=relative_player_high, shape=(2,), dtype=np.float32
            ),
            "relative_agents_positions": Box(
                low=relative_agents_low, high=relative_agents_high, shape=(self.total_enemies - 1, 2), dtype=np.float32
            ),
            "distance_to_player": Box(
                low=distance_low, high=distance_high, shape=(1,), dtype=np.float32
            ),
            "distances_to_agents": Box(
                low=np.tile(distance_low, self.total_enemies - 1),
                high=np.tile(distance_high, self.total_enemies - 1),
                shape=(self.total_enemies - 1,), dtype=np.float32
            )
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)

    def render(self):
        if self.screen is None:
            return
        self.screen.fill((0, 0, 0))
        # self.player.draw(self.screen)
        # self.screen.blit(self.player.image)

        rotated_rect = self.player.image.get_rect(center=(self.player.x + self.player.size // 2, self.player.y + self.player.size // 2))
        self.screen.blit(self.player.image, rotated_rect.topleft)

        for agent_id, agent_data in self.agent_data.items():
            pygame.draw.rect(
                self.screen,
                pygame.Color(agent_data["color"]),
                (agent_data["x"], agent_data["y"], agent_data["size"], agent_data["size"])
            )

        pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

        pass

    def reset(self, seed=None, options=None):
        # Set the random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset player and environment state
        self.player.x = self.width // 2
        self.player.y = self.height // 2
        self._initialize_enemies()

        # Reset the list of active agents to the starting agents
        self.agents = self.possible_agents[:]

        # Reset counters and other state-tracking variables
        self.frames = 0

        # Initialize observations and infos for all agents
        observations = {
            agent: {
                # Relative position to the player
                "relative_player_position": np.array([
                    self.player.x - self.agent_data[agent]["x"],
                    self.player.y - self.agent_data[agent]["y"]
                ], dtype=np.float32),

                # Relative positions to other agents (excluding itself)
                "relative_agents_positions": np.array([
                    [
                        self.agent_data[other_agent]["x"] - self.agent_data[agent]["x"],
                        self.agent_data[other_agent]["y"] - self.agent_data[agent]["y"]
                    ]
                    for other_agent in self.agents if other_agent != agent
                ], dtype=np.float32),

                # Distance to the player
                "distance_to_player": np.array([
                    np.linalg.norm([
                        self.player.x - self.agent_data[agent]["x"],
                        self.player.y - self.agent_data[agent]["y"]
                    ])
                ], dtype=np.float32),

                # Distances to other agents (excluding itself)
                "distances_to_agents": np.array([
                    np.linalg.norm([
                        self.agent_data[other_agent]["x"] - self.agent_data[agent]["x"],
                        self.agent_data[other_agent]["y"] - self.agent_data[agent]["y"]
                    ])
                    for other_agent in self.agents if other_agent != agent
                ], dtype=np.float32)
            }
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        # Update the environment state
        self.state = observations

        # Handle rendering if the environment is set to human mode
        if self.render_mode == "human" and self.screen is not None:
            self.render()

        return observations

    def step(self, actions):

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        terminate = False

        # Move player as before
        self.player.avoid_enemies2(self.agent_data, self.width, self.height)

        prev_dist = []
        curr_dist = []

        self.agent_prev_data = self.agent_data

        # Calculate distances before and after move
        for agent, information in self.agent_data.items():
            prev_distance = calculate_distance(information['x'], information['y'], self.player.x, self.player.y)
            prev_dist.append(prev_distance)

            if agent in actions:
                self._move_agent(agent, actions[agent])

            curr_distance = calculate_distance(information['x'], information['y'], self.player.x, self.player.y)
            curr_dist.append(curr_distance)

        rewards = {}
        collision_occurred = any(self._check_collision_with_player(agent) for agent in self.agents)

        for i, agent in enumerate(self.agents):
            reward = 0.0  # Initialize the reward
            distance_change = prev_dist[i] - curr_dist[i]

            # Distance-Based Reward
            reward += distance_change * 0.6
            if distance_change > 0:
                reward += 0.01

            # Proximity Bonuses
            if curr_dist[i] < 100:
                reward += 0.5
            if curr_dist[i] < 50:
                reward += 2.0
            if curr_dist[i] < 30:
                reward += 5.0

                # Track consecutive steps close to player
                # Initialize if not present
                if agent not in self.consecutive_close_steps:
                    self.consecutive_close_steps[agent] = 0
                self.consecutive_close_steps[agent] += 1

                # Every 5 steps close, give a small bonus
                if self.consecutive_close_steps[agent] % 5 == 0:
                    reward += 0.1
            else:
                # Not within 30 units, reset count
                self.consecutive_close_steps[agent] = 0

            # Penalty for Collisions with Other Agents
            if self._check_collision_with_agents(agent):
                reward -= 4.0

            # Cooperation Reward
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    distance_to_other_agent = calculate_distance(
                        self.agent_data[agent]["x"], self.agent_data[agent]["y"],
                        self.agent_data[other_agent]["x"], self.agent_data[other_agent]["y"]
                    )
                    if 20 < distance_to_other_agent < 40:
                        reward += 1.5

            # Small Step Penalty
            reward -= 0.01

            # Catch-the-Player Reward
            if collision_occurred:
                reward += 10.0
                terminate = True

            # Last action penalty for jittering
            current_action = actions[agent]
            if agent in self.last_actions:
                if self.last_actions[agent] != current_action:
                    reward -= 0.01
            self.last_actions[agent] = current_action

            # Clip reward
            rewards[agent] = np.clip(reward, -20, 20)

        terminations = {agent: terminate for agent in self.agents}

        self.frames += 1
        env_truncation = self.frames >= MAX_STEPS
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {
            agent: {
                "relative_player_position": np.array([
                    self.player.x - self.agent_data[agent]["x"],
                    self.player.y - self.agent_data[agent]["y"]
                ], dtype=np.float32),
                "relative_agents_positions": np.array([
                    [
                        self.agent_data[other_agent]["x"] - self.agent_data[agent]["x"],
                        self.agent_data[other_agent]["y"] - self.agent_data[agent]["y"]
                    ]
                    for other_agent in self.agents if other_agent != agent
                ], dtype=np.float32),
                "distance_to_player": np.array([
                    np.linalg.norm([
                        self.player.x - self.agent_data[agent]["x"],
                        self.player.y - self.agent_data[agent]["y"]
                    ])
                ], dtype=np.float32),
                "distances_to_agents": np.array([
                    np.linalg.norm([
                        self.agent_data[other_agent]["x"] - self.agent_data[agent]["x"],
                        self.agent_data[other_agent]["y"] - self.agent_data[agent]["y"]
                    ])
                    for other_agent in self.agents if other_agent != agent
                ], dtype=np.float32)
            }
            for agent in self.agents
        }

        self.state = observations
        infos = {agent: {} for agent in self.agents}

        if env_truncation or terminate:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        movement = ACTION_MAP[action]

        self.agent_data[agent]['x'] = max(0, min(self.width - self.agent_data[agent]['size'],
                                                 self.agent_data[agent]['x'] + movement[0]))
        self.agent_data[agent]['y'] = max(0, min(self.height - self.agent_data[agent]['size'],
                                                 self.agent_data[agent]['y'] + movement[1]))

    def _check_collision_with_agents(self, agent):
        agent_rect = pygame.Rect(
            self.agent_data[agent]["x"],
            self.agent_data[agent]["y"],
            self.agent_data[agent]["size"],
            self.agent_data[agent]["size"]
        )

        for other_agent, other_data in self.agent_data.items():
            if other_agent == agent:  # Skip checking collision with itself
                continue

            other_agent_rect = pygame.Rect(
                other_data["x"],
                other_data["y"],
                other_data["size"],
                other_data["size"]
            )

            if agent_rect.colliderect(other_agent_rect):
                return True  # Collision detected

        return False  # No collisions

    def _check_collision_with_player(self, agent):
        agent_rect = pygame.Rect(
            self.agent_data[agent]["x"],
            self.agent_data[agent]["y"],
            self.agent_data[agent]["size"],
            self.agent_data[agent]["size"]
        )

        player_rect = pygame.Rect(
            self.player.x,
            self.player.y,
            self.player.size,
            self.player.size
        )

        return agent_rect.colliderect(player_rect)

    def _is_near_wall(self, agent):
        x, y = self.agent_data[agent]["x"], self.agent_data[agent]["y"]
        test1 = self.width - x
        test2 = self.height - y

        return test1, test2


'''
env = NeuralDash(render_mode="human", number_of_enemies=2)

# Initialize the environment and print initial agent data
obs, infos = env.reset()

# Game loop to keep the window open
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Exit the game loop

    # Render the environment
    env.render()
    movements = {agent: env.action_space(agent).sample() for agent in env.agents}
    one, two, three, four, five = env.step(movements)
    if four['enemy_0']:
        running = False
'''
