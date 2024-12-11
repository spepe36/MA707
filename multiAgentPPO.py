import numpy as np
import torch
import torch.nn as nn
from multiAgentEnv import NeuralDash
import os
import torch.nn.utils as nn_utils
import matplotlib.pyplot as plt
import time


# Define your neural network
class AgentPolicy(nn.Module):
    def __init__(self, obs_dims, action_dims):
        super(AgentPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dims)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        # Return raw logits
        return self.fc3(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dims):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Single output: the value of the state

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No activation; raw value output


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def flatten_agent_obs(agent_obs):
    relative_player_position = agent_obs['relative_player_position'].flatten()
    relative_agents_positions = agent_obs['relative_agents_positions'].flatten()
    distance_to_player = agent_obs['distance_to_player'].flatten()
    distances_to_agents = agent_obs['distances_to_agents'].flatten()

    obs_vector = np.concatenate([
        relative_player_position,
        relative_agents_positions,
        distance_to_player,
        distances_to_agents
    ])

    return obs_vector


def compute_loss(
    policy, old_policy, critic, observations, actions, advantages, returns, clip_epsilon=0.2
):
    # Forward pass through the policy
    logits = policy(observations)
    # Stabilize logits before softmax
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
    probs = torch.exp(logits_stable)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = torch.log(probs + 1e-8)

    with torch.no_grad():
        logits_old = old_policy(observations)
        logits_old_stable = logits_old - logits_old.max(dim=-1, keepdim=True)[0]
        old_probs = torch.exp(logits_old_stable)
        old_probs = old_probs / (old_probs.sum(dim=-1, keepdim=True) + 1e-8)
        old_log_probs = torch.log(old_probs + 1e-8)

    current_log_prob_actions = torch.gather(log_probs, 1, actions.view(-1, 1)).squeeze()
    old_log_prob_actions = torch.gather(old_log_probs, 1, actions.view(-1, 1)).squeeze()

    ratios = torch.exp(current_log_prob_actions - old_log_prob_actions)
    clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)

    surrogate_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    # Correct entropy calculation
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    value_preds = critic(observations).squeeze()
    value_loss = ((value_preds - returns) ** 2).mean()

    total_loss = surrogate_loss + 0.5 * value_loss - 0.01 * entropy

    return total_loss, value_loss, surrogate_loss


def update_policy(
    policy, old_policy, critic, transitions, advantages, returns,
    optimizer_policy, optimizer_value, device, clip_epsilon=0.2, num_epochs=4, batch_size=32
):
    states = torch.tensor(np.array([flatten_agent_obs(t["state"]) for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(num_epochs):
        indices = torch.randperm(len(states))
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Compute policy loss and optimize
            total_loss, _, policy_loss = compute_loss(
                policy, old_policy, critic, batch_states, batch_actions, batch_advantages, batch_returns, clip_epsilon)
            optimizer_policy.zero_grad()
            total_loss.backward()

            # Apply gradient clipping before optimizer step
            nn_utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer_policy.step()

            # Compute value loss separately and optimize
            value_preds = critic(batch_states).squeeze()
            value_loss = ((value_preds - batch_returns) ** 2).mean()

            optimizer_value.zero_grad()
            value_loss.backward()

            # Apply gradient clipping for value network as well
            nn_utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

            optimizer_value.step()


def compute_discounted_rewards(rewards, gamma=0.99):
    if not rewards:
        raise ValueError("Empty rewards list provided to compute_discounted_rewards.")

    discounted = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted.insert(0, cumulative)
    return discounted


def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def train_ppo(num_episodes=100, rollout_steps=2000, gamma=0.99, lam=0.95, render_mode='False',
              policy_path=None, critic_path=None, name="nonmp"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = NeuralDash(render_mode=render_mode)
    obs_dims = 6
    action_dims = 5
    policy = AgentPolicy(obs_dims, action_dims).to(device)
    old_policy = AgentPolicy(obs_dims, action_dims).to(device)
    critic = ValueNetwork(obs_dims).to(device)

    # If we have saved models to continue training from:
    if policy_path is not None and critic_path is not None:
        policy_state_dict = torch.load(policy_path, map_location=device)
        critic_state_dict = torch.load(critic_path, map_location=device)

        policy.load_state_dict(policy_state_dict)
        critic.load_state_dict(critic_state_dict)
    else:
        # Starting fresh
        policy.apply(init_weights)
        critic.apply(init_weights)

    old_policy.load_state_dict(policy.state_dict())

    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=.00005)
    optimizer_value = torch.optim.Adam(critic.parameters(), lr=.00005)

    # Lists to store metrics
    episode_rewards_log = []  # Store total reward per episode

    for episode in range(num_episodes):
        states = env.reset()

        transitions = {agent: [] for agent in env.agents}
        rewards = {agent: [] for agent in env.agents}
        values = {agent: [] for agent in env.agents}
        done = {agent: False for agent in env.agents}

        episode_reward = 0.0  # Keep track of total episode reward across all agents

        for step in range(rollout_steps):
            actions = {}
            obs_tensors = {}

            for agent in env.agents:
                if not done[agent]:
                    max_pos = 700
                    flattened_state = flatten_agent_obs(states[agent])
                    normalized_state = flattened_state / max_pos

                    obs_tensor = torch.tensor(normalized_state, dtype=torch.float32)
                    logits = policy(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).item()

                    actions[agent] = action
                    obs_tensors[agent] = obs_tensor

            next_states, rewards_step, truncations, terminations, _ = env.step(actions)
            done = {agent: truncations[agent] or terminations[agent] for agent in truncations.keys()}

            for agent in env.agents:
                if not done[agent]:
                    transitions[agent].append({
                        "state": states[agent],
                        "action": actions[agent],
                        "reward": rewards_step[agent],
                        "next_state": next_states[agent],
                        "done": done[agent],
                    })
                    rewards[agent].append(rewards_step[agent])
                    values[agent].append(critic(obs_tensors[agent]).item())

                episode_reward += rewards_step[agent]  # Add agent's reward to total episode reward

            if render_mode == 'human':
                env.render()

            states = next_states

            if all(done.values()):
                break

        advantages = {}
        returns = {}
        for agent in env.agents:
            next_value = 0 if done[agent] else critic(torch.tensor(flatten_agent_obs(states[agent]), dtype=torch.float32)).item()
            advantages[agent] = compute_gae(rewards[agent], values[agent], next_value, gamma=gamma, lam=lam)
            returns[agent] = compute_discounted_rewards(rewards[agent], gamma=gamma)

        combined_transitions = []
        combined_advantages = []
        combined_returns = []

        for agent in env.agents:
            if transitions[agent]:
                combined_transitions.extend(transitions[agent])
            if advantages[agent]:
                combined_advantages.extend(advantages[agent])
            if returns[agent]:
                combined_returns.extend(returns[agent])

        update_policy(
            policy, old_policy, critic, combined_transitions, combined_advantages, combined_returns,
            optimizer_policy, optimizer_value, device=device
        )

        old_policy.load_state_dict(policy.state_dict())

        # Log the total episode reward
        episode_rewards_log.append(episode_reward)

        if episode == num_episodes // 2:
            save_model(policy, critic, name, directory="models", episode=episode)

        if episode == num_episodes - 1:
            save_model(policy, critic, name, directory="models", episode=episode)

        print(f"Episode {episode + 1}/{num_episodes} completed. Episode Reward: {episode_reward}")

    # After training is completed, create a reward curve plot
    plt.figure()
    plt.plot(episode_rewards_log)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.savefig('reward_curve.png')
    plt.close()
    print("Saved reward curve plot as reward_curve.png")

    # Action Distribution Plot
    # Pick a representative state (e.g., zero state or a known scenario)
    sample_state = np.zeros(obs_dims, dtype=np.float32)  # or use a meaningful state
    with torch.no_grad():
        obs_tensor = torch.tensor(sample_state, dtype=torch.float32, device=device)
        logits = policy(obs_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    plt.figure()
    actions = np.arange(action_dims)
    plt.bar(actions, probs)
    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Action Distribution at Sample State')
    plt.savefig('action_distribution.png')
    plt.close()
    print("Saved action distribution plot as action_distribution.png")


def visualize_ppo(policy_model, critic_model):

    policy = AgentPolicy(6, 5)
    critic = ValueNetwork(6)
    policy.load_state_dict(torch.load(f"models/{policy_model}.pth"))
    critic.load_state_dict(torch.load(f"models/{critic_model}.pth"))
    policy.eval()  # set to evaluation mode

    env = NeuralDash(render_mode='human')
    state = env.reset()

    done = {agent: False for agent in env.agents}
    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if not done[agent]:
                obs = torch.tensor(flatten_agent_obs(state[agent]), dtype=torch.float32)
                logits = policy(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()
                actions[agent] = action

        next_state, reward, truncation, termination, info = env.step(actions)
        done = {agent: truncation[agent] or termination[agent] for agent in env.agents}

        env.render()

        time.sleep(.005)
        state = next_state


def save_model(policy, critic, name, directory="models", episode=None):
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save policy and critic models
    policy_path = os.path.join(directory, f"{name}policy{'_ep' + str(episode) if episode is not None else ''}.pth")
    critic_path = os.path.join(directory, f"{name}critic{'_ep' + str(episode) if episode is not None else ''}.pth")

    torch.save(policy.state_dict(), policy_path)
    torch.save(critic.state_dict(), critic_path)

    print(f"Models saved: Policy -> {policy_path}, Critic -> {critic_path}")


if __name__ == "__main__":
    # train_ppo(num_episodes=5000, rollout_steps=1500, gamma=0.8, lam=0.95, name='nonmp_5000', policy_path='models/p_12000_sims_2agent.pth', critic_path='models/c_12000_sims_2agent.pth')
    visualize_ppo('p_1500_sims_2agents', 'c_1500_sims_2agents')
