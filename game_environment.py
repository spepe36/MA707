import pygame
import random
from game_objects import Player, Enemy
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class EnemyAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Define the device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural network setup
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # A simple neural network model for Q-learning
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert state to tensor
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()  # Return action with highest Q-value

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Convert state and next_state to tensors
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            # Predicted Q-values for the current state
            current_q_values = self.model(state)

            # Target Q-values
            with torch.no_grad():
                next_q_values = self.model(next_state)
                max_next_q_value = torch.max(next_q_values).item()
                target = reward + (self.gamma * max_next_q_value if not done else reward)

            # Update only the Q-value for the chosen action
            target_f = current_q_values.clone()
            target_f[action] = target  # Only modify the Q-value of the chosen action

            # Compute the loss only for the chosen action
            loss = self.criterion(current_q_values[action], torch.FloatTensor([target]).to(self.device))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class GameEnv:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.player = Player(width // 2, height // 2, 30, (255, 0, 0), 4, 1, width, height, 5)
        self.enemies = []
        self.spawn_interval = 600
        self.last_spawn_time = 0
        self.start_time = pygame.time.get_ticks()
        self.max_duration = 30000
        self.max_enemies = 25
        self.done = False

    def reset(self):
        self.player.x, self.player.y = self.width // 2, self.height // 2
        self.player.health = 1
        self.enemies.clear()
        self.last_spawn_time = pygame.time.get_ticks()
        self.start_time = pygame.time.get_ticks()
        self.done = False
        return self.get_state()

    def get_state(self):
        player_state = [self.player.x, self.player.y]
        enemies_state = []

        for enemy in self.enemies:
            enemies_state.extend([enemy.x, enemy.y, enemy.size])

        # Ensure a fixed length state even if there are no enemies
        max_enemies = self.max_enemies
        while len(enemies_state) < max_enemies * 3:
            enemies_state.extend([0, 0, 0])  # Padding with zeros

        return player_state + enemies_state

    def step(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy()
            self.last_spawn_time = current_time

        reward, self.done = self.check_collisions()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= self.max_duration:
            reward += 10
            self.done = True

        state = self.get_state()
        return state, reward, self.done

    def spawn_enemy(self):
        """Function to spawn enemies at a given location."""
        enemy_information = {
            'yellow': [30, 2.6, (255, 255, 0)],  # [Size, speed, color]
            'green': [15, 3.0, (0, 255, 0)],
            'red': [60, 2, (255, 0, 0)],
            'purple': [40, 3.3, (255, 0, 255)],
        }

        if len(self.enemies) >= self.max_enemies:
            return None

        edge = random.choice(['top', 'bottom', 'left', 'right'])

        # Always choose 'yellow' enemy for the environment
        enemy_color = 'yellow'
        enemy_size = enemy_information[enemy_color][0]
        enemy_speed = enemy_information[enemy_color][1]
        enemy_color = enemy_information[enemy_color][2]

        x, y = 0, 0

        if edge == 'top':
            x = random.randint(0, self.width - enemy_size)
            y = 0
        elif edge == 'bottom':
            x = random.randint(0, self.width - enemy_size)
            y = self.height - enemy_size
        elif edge == 'left':
            x = 0
            y = random.randint(0, self.height - enemy_size)
        elif edge == 'right':
            x = self.width - enemy_size
            y = random.randint(0, self.height - enemy_size)

        new_enemy = Enemy(x, y, enemy_size, enemy_color, enemy_speed)
        self.enemies.append(new_enemy)

    def check_collisions(self):
        player_rect = (self.player.x, self.player.y, self.player.size, self.player.size)
        for enemy in self.enemies:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if pygame.Rect(player_rect).colliderect(pygame.Rect(enemy_rect)):
                return -10, True
        return 0, False

    def render(self):
        self.screen.fill((230, 230, 230))
        self.player.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


def main():
    width = 1200
    height = 900
    game_env = GameEnv(width, height)
    state_size = len(game_env.get_state())
    action_size = 5  # UP, DOWN, LEFT, RIGHT, STAY
    enemy_agent = EnemyAgent(state_size, action_size)
    batch_size = 32

    while True:
        state = game_env.reset()
        done = False

        while not done:
            # Use the Player AI to decide the player's action
            player_action = game_env.player.avoid_enemies(game_env.enemies, width, height)
            if not player_action:
                player_action = "STAY"

            # Initialize next_state to ensure it's always defined
            next_state = state  # Default to current state if no action is taken
            reward = 0  # Default reward if no action is taken

            next_state, reward, done = game_env.step()

            # Enemy actions based on the RL agent
            actions = []
            for enemy in game_env.enemies:
                enemy_action = enemy_agent.act(state)
                actions.append(enemy_action)

            # Apply actions to the enemies
            for idx, enemy_action in enumerate(actions):
                if enemy_action == 0:
                    game_env.enemies[idx].y -= game_env.enemies[idx].speed  # UP
                elif enemy_action == 1:
                    game_env.enemies[idx].y += game_env.enemies[idx].speed  # DOWN
                elif enemy_action == 2:
                    game_env.enemies[idx].x -= game_env.enemies[idx].speed  # LEFT
                elif enemy_action == 3:
                    game_env.enemies[idx].x += game_env.enemies[idx].speed  # RIGHT
                # Action 4 means STAY, no movement

            # Render the game
            game_env.render()

            # Store the experience for each enemy in the agent's memory
            for idx, enemy in enumerate(game_env.enemies):
                # Remember the action taken by each enemy
                enemy_agent.remember(state, actions[idx], reward, next_state, done)

            # Update state to the next state for the next iteration
            state = next_state

            # Train the enemy's RL agent
            enemy_agent.replay(batch_size)

        print("Game Over!")
        pygame.time.delay(2000)


if __name__ == "__main__":
    main()