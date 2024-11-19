import pygame
import random
from game_objects import Player, Enemy
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pickle
import time
import math


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


# Define the EnemyAgent class
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

        # Sample a batch from the memory
        minibatch = random.sample(self.memory, batch_size)

        # Convert the minibatch into tensors
        states = torch.FloatTensor([item[0] for item in minibatch]).to(self.device)
        actions = torch.LongTensor([item[1] for item in minibatch]).to(self.device)
        rewards = torch.FloatTensor([item[2] for item in minibatch]).to(self.device)
        next_states = torch.FloatTensor([item[3] for item in minibatch]).to(self.device)
        dones = torch.FloatTensor([item[4] for item in minibatch]).to(self.device)

        # Predicted Q-values for the current states
        current_q_values = self.model(states)

        # Predicted Q-values for the next states
        with torch.no_grad():
            next_q_values = self.model(next_states)

        # Target Q-value calculation
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Only update the Q-values of the chosen actions
        target_q_values = current_q_values.clone()
        target_q_values[range(batch_size), actions] = targets

        # Compute the loss for the batch
        loss = self.criterion(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class GameEnv:
    def __init__(self, width, height, visual_mode=True):
        self.visual_mode = visual_mode
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.player = Player(width // 2, height // 2, 30, (255, 0, 0), 4, 1, width, height, 5)
        self.enemies = []
        self.spawn_interval = 600
        self.last_spawn_time = 0
        self.start_time = pygame.time.get_ticks()
        self.max_duration = 30000
        self.max_enemies = 25
        self.done = False

        # Initialize pygame if in visual mode
        if self.visual_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
        else:
            self.screen = None  # No screen if not in visual mode

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

        # Optimize by using list comprehension and reusing space for state representation
        for enemy in self.enemies:
            enemies_state.extend([enemy.x, enemy.y, enemy.size, enemy.angle_to_player])

        max_enemies = self.max_enemies
        # Remove redundant zero-padding logic
        enemies_state = enemies_state[:max_enemies * 4]  # Crop if more than needed
        while len(enemies_state) < max_enemies * 4:
            enemies_state.extend([0, 0, 0, 0])  # Pad only when necessary

        return player_state + enemies_state

    def step(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy()
            self.last_spawn_time = current_time

        previous_distance = self.calculate_distance_to_player()

        # Efficiently update enemies using a single loop
        for enemy in self.enemies:
            enemy.update(self.player.x, self.player.y)

        reward, self.done = self.calculate_reward(previous_distance)

        # Reward calculation should be handled once, after all updates
        elapsed_time = current_time - self.start_time
        if elapsed_time >= self.max_duration:
            reward -= 100
            self.done = True

        state = self.get_state()
        return state, reward, self.done

    def spawn_enemy(self):
        if len(self.enemies) >= self.max_enemies:
            return None

        enemy_color = 'yellow'
        enemy_size = 30
        enemy_speed = 2.6

        edge = random.choice(['top', 'bottom', 'left', 'right'])
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

    def calculate_distance_to_player(self):
        if not self.enemies:
            return float('inf')

        distances = [math.hypot(enemy.x - self.player.x, enemy.y - self.player.y) for enemy in self.enemies]
        return min(distances)

    def calculate_reward(self, previous_distance):
        current_distance = self.calculate_distance_to_player()
        collision_with_player = self.check_collision_with_player()

        # Reward function
        if collision_with_player:
            return 1000, True
        elif current_distance < previous_distance:
            return 20, False
        elif current_distance > previous_distance:
            return -50, False
        else:
            return -0.5, False

    def check_collision_with_player(self):
        player_rect = (self.player.x, self.player.y, self.player.size, self.player.size)
        for enemy in self.enemies:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if pygame.Rect(player_rect).colliderect(pygame.Rect(enemy_rect)):
                return True
        return False

    def render(self):
        if self.visual_mode and self.screen:
            self.screen.fill((0, 0, 0))
            self.player.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            pygame.display.flip()

    def close(self):
        pygame.quit()


def test_enemy_behavior():
    # Load the trained agent
    with open("trained_enemy_ai.pkl", 'rb') as f:
        enemy_agent = pickle.load(f)

    # Initialize the game environment without visual mode
    game_env = GameEnv(1200, 900, visual_mode=False)
    state = game_env.reset()

    done = False
    while not done:
        # Get the action for each enemy from the trained agent
        actions = [enemy_agent.act(state) for _ in game_env.enemies]

        # Apply the actions to each enemy
        for idx, enemy_action in enumerate(actions):
            speed = game_env.enemies[idx].speed
            if enemy_action == 0:
                game_env.enemies[idx].y -= speed  # UP
            elif enemy_action == 1:
                game_env.enemies[idx].y += speed  # DOWN
            elif enemy_action == 2:
                game_env.enemies[idx].x -= speed  # LEFT
            elif enemy_action == 3:
                game_env.enemies[idx].x += speed  # RIGHT
            # Action 4 means STAY, no movement

        # Step the environment forward
        next_state, reward, done = game_env.step()

        # Update the state for the next step
        state = next_state

        # Render if visual mode is enabled
        if game_env.visual_mode:
            game_env.render()

    game_env.close()


def test_enemy_behavior(file_name):
    # Load the trained agent
    with open(file_name, 'rb') as f:
        enemy_agent = pickle.load(f)

    # Initialize the game environment with visual mode enabled
    game_env = GameEnv(1200, 900, visual_mode=True)  # Change visual_mode to True
    state = game_env.reset()

    done = False
    while not done:
        # Get the action for each enemy from the trained agent
        actions = [enemy_agent.act(state) for _ in game_env.enemies]

        # Apply the actions to each enemy
        for idx, enemy_action in enumerate(actions):
            speed = game_env.enemies[idx].speed
            if enemy_action == 0:
                game_env.enemies[idx].y -= speed  # UP
            elif enemy_action == 1:
                game_env.enemies[idx].y += speed  # DOWN
            elif enemy_action == 2:
                game_env.enemies[idx].x -= speed  # LEFT
            elif enemy_action == 3:
                game_env.enemies[idx].x += speed  # RIGHT
            # Action 4 means STAY, no movement

        # Step the environment forward
        next_state, reward, done = game_env.step()

        # Update the state for the next step
        state = next_state

        # Render if visual mode is enabled
        if game_env.visual_mode:
            game_env.render()

    game_env.close()


def run_fast_simulations(num_simulations=100, visual_mode=False, save_path="enemy_ai.pkl"):
    width = 1200
    height = 900
    game_env = GameEnv(width, height, visual_mode)
    state_size = len(game_env.get_state())
    action_size = 5  # UP, DOWN, LEFT, RIGHT, STAY
    enemy_agent = EnemyAgent(state_size, action_size)
    batch_size = 32

    for simulation in range(num_simulations):
        print(f"Starting simulation {simulation + 1}/{num_simulations}")

        state = game_env.reset()
        done = False
        simulation_time = 0  # Track time step in simulation (no reliance on real-time)

        while not done:
            # Instead of using real time, control the steps manually
            # Adjust the amount of time progression per step (e.g., simulate faster time)
            simulation_time += 1  # Each loop counts as a simulation step

            player_action = game_env.player.avoid_enemies(game_env.enemies, width, height)
            if not player_action:
                player_action = "STAY"

            next_state, reward, done = game_env.step()

            actions = [enemy_agent.act(state) for _ in game_env.enemies]

            for idx, enemy_action in enumerate(actions):
                speed = game_env.enemies[idx].speed
                if enemy_action == 0:
                    game_env.enemies[idx].y -= speed  # UP
                elif enemy_action == 1:
                    game_env.enemies[idx].y += speed  # DOWN
                elif enemy_action == 2:
                    game_env.enemies[idx].x -= speed  # LEFT
                elif enemy_action == 3:
                    game_env.enemies[idx].x += speed  # RIGHT
                # Action 4 means STAY, no movement

            # Skip rendering if visual_mode is False
            if visual_mode:
                game_env.render()
                game_env.clock.tick(60)  # Limit FPS to 60

            # Memory storage timing
            for idx, enemy in enumerate(game_env.enemies):
                enemy_agent.remember(state, actions[idx], reward, next_state, done)

            # Update state to the next state for the next iteration
            state = next_state

            # Training (replay) timing
            enemy_agent.replay(batch_size)

            # Check if we should end the simulation based on our simulated time, not real-time
            if simulation_time >= game_env.max_duration // 10:  # Speed up by a factor of 10
                reward -= 10
                done = True

    # Save the trained enemy AI after all simulations
    with open(save_path, 'wb') as f:
        pickle.dump(enemy_agent, f)
    print(f"Trained enemy AI saved to {save_path}.")


if __name__ == "__main__":

    run_fast_simulations(num_simulations=10000, visual_mode=False, save_path="trained_enemy_ai.pkl")

    test_enemy_behavior("trained_enemy_ai.pkl")
