import pygame
from game_objects import Player, Enemy
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pickle
import math
import matplotlib.pyplot as plt
import cProfile
import pstats
from io import StringIO


# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(152, 128)  # Adjust in_features to match input size
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

        # DQN
        self.model = DQN(state_size, action_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)

        # Adjusted to the updated state structure
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
    def __init__(self, width, height, enemy_agent, visual_mode=True):
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
        self.simulated_time = 0
        self.time_step = 50
        self.done = False
        self.enemy_agent = enemy_agent

        # Initialize pygame if in visual mode
        if self.visual_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
        else:
            self.screen = None  # No screen if not in visual mode

    def action_to_movement(self, action, enemy):
        # Define possible actions: UP, DOWN, LEFT, RIGHT, STAY
        # Each action corresponds to a movement direction in terms of (dx, dy)

        if action == 0:  # UP
            dx, dy = 0, -enemy.velocity.y
        elif action == 1:  # DOWN
            dx, dy = 0, enemy.velocity.y
        elif action == 2:  # LEFT
            dx, dy = -enemy.velocity.x, 0
        elif action == 3:  # RIGHT
            dx, dy = enemy.velocity.x, 0
        else:  # STAY (no movement)
            dx, dy = 0, 0

        return dx, dy

    def reset(self):
        self.player.x, self.player.y = self.width // 2, self.height // 2
        self.simulated_time = 0
        self.last_spawn_time = 0
        self.player.health = 1
        self.enemies.clear()
        self.start_time = pygame.time.get_ticks()
        self.done = False
        return self.get_state()

    def get_state(self):
        player_state = [self.player.x / self.width, self.player.y / self.height]  # Normalized player position
        enemies_state = []

        max_distance = math.sqrt(self.width ** 2 + self.height ** 2)

        for enemy in self.enemies:
            # Normalize enemy state values as before
            normalized_x = enemy.x / self.width
            normalized_y = enemy.y / self.height
            normalized_angle = (enemy.angle_to_player + math.pi) / (2 * math.pi)
            normalized_vx = enemy.velocity.x / 2.6
            normalized_vy = enemy.velocity.y / 2.6

            direction_to_player = pygame.Vector2(self.player.x - enemy.x, self.player.y - enemy.y)
            dist_to_player = direction_to_player.length()
            normalized_distance = dist_to_player / max_distance

            enemies_state.extend([normalized_x, normalized_y, normalized_angle,
                                  normalized_vx, normalized_vy, normalized_distance])

        max_enemies = self.max_enemies
        enemies_state = enemies_state[:max_enemies * 6]

        while len(enemies_state) < max_enemies * 6:
            enemies_state.extend([0, 0, 0, 0, 0, 0])

        return player_state + enemies_state

    def step(self):
        self.simulated_time += self.time_step  # Advance simulated time

        if self.simulated_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy()
            self.last_spawn_time = self.simulated_time

        state = self.get_state()

        # Store rewards and actions for each enemy
        rewards = []
        actions = []
        previous_distances = []

        # Loop through each enemy, calculate its distance to the player, and assign individual rewards
        for enemy in self.enemies:
            previous_distance = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)
            previous_distances.append(previous_distance)

            # Action for each enemy (could be from the agent or random)
            action = self.enemy_agent.act(state)
            actions.append(action)

            # Update each enemy's position based on its action
            dx, dy = self.action_to_movement(action, enemy)
            enemy.x += dx
            enemy.y += dy

            # Calculate the new distance to player
            current_distance = math.hypot(enemy.x - self.player.x, enemy.y - self.player.y)

            # Calculate the reward for this enemy based on the change in distance
            reward, done = self.calculate_reward(previous_distance, current_distance)
            if done:
                self.done = True
            rewards.append(reward)

        return state, rewards, self.done

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

    def calculate_reward(self, previous_distance, current_distance):
        collision_with_player = self.check_collision_with_player()

        # Reward function
        if collision_with_player:
            return 10, True
        elif current_distance < previous_distance:
            return 1, False  # Reward for getting closer to the player
        elif current_distance > previous_distance:
            return -1, False  # Penalty for moving away from the player
        else:
            return -1, False  # Small penalty for no movement change

    def check_collision_with_player(self):
        player_rect = (self.player.x, self.player.y, self.player.size, self.player.size)
        for enemy in self.enemies:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if pygame.Rect(player_rect).colliderect(pygame.Rect(enemy_rect)):
                return True
        return False

    def render(self):
        if not self.visual_mode or not self.screen:
            return  # Skip rendering in non-visual mode

        self.screen.fill((0, 0, 0))
        self.player.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        pygame.display.flip()

    @staticmethod
    def close():
        pygame.quit()


# Run when checking results of the model
def test_enemy_behavior(file_name):
    width, height = 1200, 900

    # Load the trained agent
    with open(file_name, 'rb') as f:
        enemy_agent = pickle.load(f)

    # Initialize the game environment with visual mode enabled
    game_env = GameEnv(width, height, enemy_agent, visual_mode=True)  # Change visual_mode to True
    state = game_env.reset()

    done = False
    while not done:
        # Get the action for each enemy from the trained agent
        actions = [enemy_agent.act(state) for _ in game_env.enemies]

        player_action = game_env.player.avoid_enemies(game_env.enemies, width, height)
        if not player_action:
            player_action = "STAY"

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
        next_state, rewards, done = game_env.step()

        # Update the state for the next step
        state = next_state

        # Render if visual mode is enabled
        if game_env.visual_mode:
            game_env.render()

    game_env.close()


def plot_rewards(rewards_per_simulation):
    plt.plot(range(1, len(rewards_per_simulation) + 1), rewards_per_simulation)
    plt.xlabel('Simulation Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Simulation Episode')
    plt.show()


def run_simulations(num_simulations=100, visual_mode=False, save_path="enemy_ai.pkl"):
    width, height = 1200, 900

    temporary_enemy_agent = EnemyAgent(state_size=0, action_size=5)  # Default state size, will update later
    game_env = GameEnv(width, height, temporary_enemy_agent, visual_mode)
    state_size = len(game_env.get_state())
    enemy_agent = EnemyAgent(state_size, action_size=5)
    game_env = GameEnv(width, height, enemy_agent, visual_mode)

    batch_size = 64
    time_scaling_factor = 10
    rewards_per_simulation = []  # Store the total rewards for each simulation

    for simulation in range(num_simulations):
        print(f"Starting simulation {simulation + 1}/{num_simulations}")

        state = game_env.reset()
        done = False
        simulation_time = 0
        total_rewards = 0  # Initialize total rewards for this simulation

        while not done:
            simulation_time += 1  # Increment simulated time

            # Player action
            game_env.player.avoid_enemies(game_env.enemies, width, height) or "STAY"

            # Environment step
            next_state, rewards, done = game_env.step()

            total_rewards += sum(rewards)  # Accumulate rewards for this step

            # Enemy actions
            actions = [enemy_agent.act(state) for _ in game_env.enemies]
            for enemy, action in zip(game_env.enemies, actions):
                dx, dy = game_env.action_to_movement(action, enemy)
                enemy.x += dx
                enemy.y += dy

            # Memory storage
            for idx, enemy in enumerate(game_env.enemies):
                enemy_agent.remember(state, actions[idx], rewards[idx], next_state, done)

            state = next_state

            # End simulation early if time exceeds scaled duration
            if simulation_time >= game_env.max_duration // time_scaling_factor:
                done = True

        # Save the total rewards for this simulation
        rewards_per_simulation.append(total_rewards)

        # Train the agent after all simulations
        if len(enemy_agent.memory) >= batch_size:
            enemy_agent.replay(batch_size)

    # Save the trained enemy AI after all simulations
    with open(save_path, 'wb') as f:
        pickle.dump(enemy_agent, f)
    print(f"Trained enemy AI saved to {save_path}.")

    # Return the rewards for further analysis
    return rewards_per_simulation


def profile_run_simulations(*args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    result = run_simulations(*args, **kwargs)
    profiler.disable()  # Stop profiling

    # Print or save the profiling results
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumtime')  # Sort by cumulative time
    stats.print_stats()  # Print the profiling results to the console
    print(s.getvalue())  # Output the profiling results

    return result


if __name__ == "__main__":

    # rewards = run_simulations(num_simulations=250, visual_mode=False, save_path="trained_enemy_ai_250sims.pkl")
    # profile_run_simulations(num_simulations=100, visual_mode=False, save_path="enemy_ai.pkl")

    # plot_rewards(rewards)

    # Run code below to see result of the simulations:

    test_enemy_behavior("enemy_ai.pkl")


