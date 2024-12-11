import functools
import pygame
import sys
from gymnasium.spaces import Discrete, Dict, Box
import numpy as np
from pettingzoo import ParallelEnv
from game_objects import Player
import random

MAX_STEPS = 5000
ACTION_MAP = {
    0: (0, -2.6),  # Up
    1: (0, 2.6),   # Down
    2: (-2.6, 0),  # Left
    3: (2.6, 0),   # Right
    4: (0, 0)       # Stay
}

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class NeuralDashAi(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, player_speed=4, env_width=700, env_height=700, number_of_enemies=2, render_mode=None):
        self.state = None
        self.total_enemies = number_of_enemies
        self.width = env_width
        self.height = env_height
        self.possible_agents = ["enemy_" + str(r) for r in range(number_of_enemies)]
        self.frames = 0
        self.last_actions = {}
        self.consecutive_close_steps = {}

        self.player = Player(env_width // 2, env_height // 2, 30, (255, 0, 0), player_speed, 1, env_width, env_height, 5, 'player_image.png')

        self.render_mode = render_mode
        pygame.init()  # Ensure pygame is initialized before font or display usage

        # Initialize scoring
        self.start_time = pygame.time.get_ticks()
        self.score = 0
        self.font = pygame.font.SysFont(None, 30)  # font for score and other texts

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.image = pygame.image.load("enemy_image.png").convert_alpha()
        else:
            self.screen = None
            self.image = pygame.image.load("enemy_image.png").convert_alpha()

        self._initialize_enemies()

        # Fonts for start and end screens
        self.title_font = pygame.font.SysFont(None, 72)
        self.button_font = pygame.font.SysFont(None, 36)

    def _initialize_enemies(self):
        self.agent_data = {}
        for name in self.possible_agents:
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

            original_image = pygame.transform.scale(self.image, (enemy_size, enemy_size))
            # Store both original and current image for rotation
            self.agent_data[name] = {
                "x": float(x),
                "y": float(y),
                "size": enemy_size,
                "original_image": original_image,
                "image": original_image
            }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        max_position = np.array([self.width, self.height])
        min_position = -max_position

        relative_player_low = -max_position
        relative_player_high = max_position

        relative_agents_low = np.tile(min_position, (self.total_enemies - 1, 1))
        relative_agents_high = np.tile(max_position, (self.total_enemies - 1, 1))

        distance_low = np.array([0])
        distance_high = np.array([np.linalg.norm(max_position)])

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

    def show_start_screen(self):
        """Display a start screen and wait for the player to start."""
        if self.screen is None:
            return

        waiting = True
        while waiting:
            self.screen.fill((230, 230, 230))

            # Title
            title_text = self.title_font.render("Purrsuit", True, (255, 100, 100))
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 4))
            self.screen.blit(title_text, title_rect)

            # Start button
            button_width, button_height = 200, 60
            button_x = (self.width - button_width) // 2
            button_y = self.height // 1.5
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, (100, 100, 200), button_rect)

            button_text = self.button_font.render("Start Game", True, (255, 255, 255))
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, button_text_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):
                        waiting = False
                elif event.type == pygame.KEYDOWN:
                    # Pressing any key also starts the game
                    waiting = False

    def show_game_over_screen(self):
        """Display a game over screen and wait for restart or quit."""
        if self.screen is None:
            return "quit"

        while True:
            self.screen.fill((230, 230, 230))

            # Game Over text
            title_text = self.title_font.render("Game Over", True, (255, 0, 0))
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 3))
            self.screen.blit(title_text, title_rect)

            # Restart button
            button_width, button_height = 200, 60
            button_x = (self.width - button_width) // 2
            button_y = self.height // 2
            restart_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            pygame.draw.rect(self.screen, (0, 200, 0), restart_button_rect)
            restart_text = self.button_font.render("Restart", True, (255, 255, 255))
            restart_text_rect = restart_text.get_rect(center=restart_button_rect.center)
            self.screen.blit(restart_text, restart_text_rect)

            # Quit button
            quit_button_rect = pygame.Rect(button_x, button_y + 80, button_width, button_height)
            pygame.draw.rect(self.screen, (200, 0, 0), quit_button_rect)
            quit_text = self.button_font.render("Quit", True, (255, 255, 255))
            quit_text_rect = quit_text.get_rect(center=quit_button_rect.center)
            self.screen.blit(quit_text, quit_text_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button_rect.collidepoint(event.pos):
                        return "restart"
                    elif quit_button_rect.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()

    def render(self):
        if self.screen is None:
            return

        self.screen.fill((255, 255, 255))

        # Update score
        elapsed_ms = pygame.time.get_ticks() - self.start_time
        self.score = elapsed_ms // 1000

        # Draw player
        rotated_rect = self.player.image.get_rect(center=(self.player.x + self.player.size // 2, self.player.y + self.player.size // 2))
        self.screen.blit(self.player.image, rotated_rect.topleft)

        # Draw enemies
        for agent_id, agent_data in self.agent_data.items():
            self.screen.blit(agent_data["image"], (agent_data["x"], agent_data["y"]))

        # Score text
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def reset(self, seed=7, options=None):
        # Show start screen before resetting if desired
        if self.render_mode == "human":
            self.show_start_screen()

        if seed is not None:
            np.random.seed(seed)

        self.player.x = self.width // 2
        self.player.y = self.height // 2
        self._initialize_enemies()

        self.agents = self.possible_agents[:]
        self.frames = 0
        self.start_time = pygame.time.get_ticks()
        self.score = 0

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

        if self.render_mode == "human" and self.screen is not None:
            self.render()

        return observations

    def step(self, actions, player_ai=True):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        terminate = False
        pygame.event.pump()

        if player_ai:
            self.player.avoid_enemies2(self.agent_data, self.width, self.height)
        else:
            keys = pygame.key.get_pressed()
            self.player.move(keys, self.width, self.height)

        prev_dist = []
        curr_dist = []
        self.agent_prev_data = self.agent_data

        for agent, information in self.agent_data.items():
            prev_distance = calculate_distance(information['x'], information['y'], self.player.x, self.player.y)
            prev_dist.append(prev_distance)

            if agent in actions:
                self._move_agent(agent, actions[agent])

            curr_distance = calculate_distance(information['x'], information['y'], self.player.x, self.player.y)
            curr_dist.append(curr_distance)

        collision_occurred = any(self._check_collision_with_player(agent) for agent in self.agents)
        rewards = {}

        for i, agent in enumerate(self.agents):
            reward = 0.0
            distance_change = prev_dist[i] - curr_dist[i]

            reward += distance_change * 0.6
            if distance_change > 0:
                reward += 0.01

            if curr_dist[i] < 100:
                reward += 0.5
            if curr_dist[i] < 50:
                reward += 2.0
            if curr_dist[i] < 30:
                reward += 5.0
                if agent not in self.consecutive_close_steps:
                    self.consecutive_close_steps[agent] = 0
                self.consecutive_close_steps[agent] += 1
                if self.consecutive_close_steps[agent] % 5 == 0:
                    reward += 0.1
            else:
                self.consecutive_close_steps[agent] = 0

            if self._check_collision_with_agents(agent):
                reward -= 4.0

            for j, other_agent in enumerate(self.agents):
                if i != j:
                    distance_to_other_agent = calculate_distance(
                        self.agent_data[agent]["x"], self.agent_data[agent]["y"],
                        self.agent_data[other_agent]["x"], self.agent_data[other_agent]["y"]
                    )
                    if 20 < distance_to_other_agent < 40:
                        reward += 1.5

            reward -= 0.01

            if collision_occurred:
                reward += 10.0
                terminate = True

            current_action = actions[agent]
            if agent in self.last_actions:
                if self.last_actions[agent] != current_action:
                    reward -= 0.01
            self.last_actions[agent] = current_action

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
            # Show game over screen
            if self.render_mode == "human":
                action = self.show_game_over_screen()
                if action == "restart":
                    obs = self.reset()
                    return obs, rewards, {agent: False for agent in self.agents}, {agent: False for agent in self.agents}, infos
                else:
                    self.agents = []
                    return observations, rewards, terminations, truncations, infos
            else:
                self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _move_agent(self, agent, action):
        movement = ACTION_MAP[action]
        old_x, old_y = self.agent_data[agent]['x'], self.agent_data[agent]['y']
        self.agent_data[agent]['x'] = max(0, min(self.width - self.agent_data[agent]['size'],
                                                 self.agent_data[agent]['x'] + movement[0]))
        self.agent_data[agent]['y'] = max(0, min(self.height - self.agent_data[agent]['size'],
                                                 self.agent_data[agent]['y'] + movement[1]))

        dx = self.agent_data[agent]['x'] - old_x
        dy = self.agent_data[agent]['y'] - old_y
        if dx != 0 or dy != 0:
            movement_vec = pygame.math.Vector2(dx, dy)
            # Assuming enemy sprite faces up initially, use (0, -1) as baseline
            angle = movement_vec.angle_to(pygame.math.Vector2(0, -1))
            self.agent_data[agent]['image'] = pygame.transform.rotate(self.agent_data[agent]['original_image'], angle)

    def _check_collision_with_agents(self, agent):
        agent_rect = pygame.Rect(
            self.agent_data[agent]["x"],
            self.agent_data[agent]["y"],
            self.agent_data[agent]["size"],
            self.agent_data[agent]["size"]
        )

        for other_agent, other_data in self.agent_data.items():
            if other_agent == agent:
                continue

            other_agent_rect = pygame.Rect(
                other_data["x"],
                other_data["y"],
                other_data["size"],
                other_data["size"]
            )
            if agent_rect.colliderect(other_agent_rect):
                return True
        return False

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
